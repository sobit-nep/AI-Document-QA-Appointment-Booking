import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, initialize_agent, AgentType
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os
import datetime
import re
import validators
import json
from pathlib import Path
import smtplib

# Loading environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
email_address = os.getenv("EMAIL_ADDRESS")
email_password = os.getenv("EMAIL_PASSWORD")

if not api_key:
    st.error("Google API Key is missing. Please configure it in your .env file.")
    st.stop()

# Configuring Google Generative AI
from google.generativeai import configure
configure(api_key=api_key)

class DocumentProcessor:
    @staticmethod
    def get_pdf_text(pdf_docs):
        try:
            text = ""
            for pdf in pdf_docs:
                reader = PdfReader(pdf)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            st.error(f"Error reading PDF files: {e}")
            return ""

    @staticmethod
    def get_text_chunks(text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        return splitter.split_text(text)

    @staticmethod
    def get_vector_store(text_chunks):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return True
        except Exception as e:
            st.error(f"Error creating vector store: {e}")
            return False

class ChatTools:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
    def validate_email(self, email: str) -> bool:
        return validators.email(email.strip())
    
    def validate_phone(self, phone: str) -> bool:
        phone = phone.strip().replace(" ", "")
        return bool(re.match(r'^\+?\d{9,15}$', phone))
    
    def parse_date(self, query: str) -> str:
        try:
            prompt = f"""Today is {datetime.datetime.now().strftime('%Y-%m-%d')}.
            Convert this date query: "{query}" into YYYY-MM-DD format.
            Return only the date, nothing else."""
            return self.llm.predict(prompt).strip()
        except:
            return "Invalid date format"
    
    def send_email_confirmation(self, details: str) -> str:
        try:
            data = json.loads(details)
            if not all([email_address, email_password]):
                return "Email configuration missing"

            msg = MIMEMultipart()
            msg['From'] = email_address
            msg['To'] = data['email']
            msg['Subject'] = data['subject']
            msg.attach(MIMEText(data['body'], 'plain'))

            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(email_address, email_password)
                server.send_message(msg)
            return "Confirmation email sent successfully"
        except Exception as e:
            return f"Error sending email: {str(e)}"

    def save_appointment(self, details: dict) -> bool:
        try:
            file_path = Path("appointments.json")
            
            if file_path.exists():
                with open(file_path, "r") as file:
                    appointments = json.load(file)
            else:
                appointments = []

            appointments.append(details)

            with open(file_path, "w") as file:
                json.dump(appointments, file, indent=4)
            
            return True
        except Exception as e:
            st.error(f"Error saving appointment: {e}")
            return False

class ChatbotUI:
    def __init__(self):
        self.doc_processor = DocumentProcessor()
        self.tools = ChatTools()
        self.setup_agent()
        self.setup_qa_chain()

    def setup_agent(self):
        tools = [
            Tool(name="validate_email", func=self.tools.validate_email,
                 description="Validate email address"),
            Tool(name="validate_phone", func=self.tools.validate_phone,
                 description="Validate phone number"),
            Tool(name="parse_date", func=self.tools.parse_date,
                 description="Convert date query to YYYY-MM-DD format"),
            Tool(name="send_confirmation", func=self.tools.send_email_confirmation,
                 description="Send confirmation email. Input should be a JSON string with 'email', 'subject', and 'body'.")
        ]

        self.agent = initialize_agent(
            tools,
            self.tools.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def setup_qa_chain(self):
        prompt_template = """
        Answer the question as detailed as possible from the provided context. If the answer is not in
        the provided context, say: "Answer is not available in the context." Do not guess answers.
        Context:\n {context}\n
        Question:\n {question}\n
        Answer:
        """
        self.prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    def user_query(self, user_question):
        try:
            # Check if vector store exists
            if not Path("faiss_index").exists():
                return "Please upload and process a PDF document first."
            
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = db.similarity_search(user_question)
            
            chain = load_qa_chain(self.llm, chain_type="stuff", prompt=self.prompt)
            response = chain(
                {"input_documents": docs, "question": user_question}, 
                return_only_outputs=True
            )
            return response["output_text"]
        except Exception as e:
            return f"Error processing your question: {str(e)}"

    def handle_appointment(self, name, phone, email, date_query):
        try:
            parsed_date = self.tools.parse_date(date_query)
            if "Invalid" in parsed_date:
                return {"success": False, "message": "Invalid date format"}

            # Prepare email details
            email_details = json.dumps({
                "email": email,
                "subject": "Appointment Confirmation",
                "body": f"Dear {name},\n\nThis email confirms your appointment on {parsed_date}.\n\nThank you,\nThe Appointment Team"
            })

            # Use agent to validate and process appointment
            prompt = f"""Process this appointment:
            Name: {name}
            Email: {email}
            Phone: {phone}
            Date: {parsed_date}

            Steps:
            1. Validate the email
            2. Validate the phone number
            3. Send confirmation email with the following details: {email_details}"""

            response = self.agent.run(prompt)
            
            # Save appointment if everything is successful
            if "error" not in response.lower():
                appointment_details = {
                    "name": name,
                    "phone": phone,
                    "email": email,
                    "appointment_date": parsed_date
                }
                if self.tools.save_appointment(appointment_details):
                    return {"success": True, "message": f"Appointment booked successfully for {parsed_date}"}
            
            return {"success": False, "message": response}
        except Exception as e:
            return {"success": False, "message": str(e)}

def main():
    st.set_page_config(page_title="AI Chatbot with Appointment Booking", layout="wide")
    st.header("Document Chatbot with Appointment Booking 💬📅")

    # Initialize chatbot UI
    chatbot = ChatbotUI()

    # Sidebar for PDF processing
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = chatbot.doc_processor.get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = chatbot.doc_processor.get_text_chunks(raw_text)
                        if chatbot.doc_processor.get_vector_store(text_chunks):
                            st.success("Documents processed successfully!")
            else:
                st.error("Please upload PDF documents.")

    # Main chat interface
    user_question = st.text_input("Ask a Question:")
    if user_question:
        if any(keyword in user_question.lower() for keyword in ["call me", "appointment", "schedule", "book"]):
            # Appointment booking flow
            st.write("Let's schedule your appointment!")
            name = st.text_input("Name:")
            phone = st.text_input("Phone Number:")
            email = st.text_input("Email:")
            date_query = st.text_input("When would you like to schedule? (e.g., 'next Monday')")

            if st.button("Book Appointment"):
                if all([name, phone, email, date_query]):
                    with st.spinner("Processing appointment..."):
                        result = chatbot.handle_appointment(name, phone, email, date_query)
                        if result["success"]:
                            st.success(result["message"])
                        else:
                            st.error(result["message"])
                else:
                    st.error("Please fill in all fields.")
        else:
            # PDF query flow
            with st.spinner("Processing your question..."):
                response = chatbot.user_query(user_question)
                st.write(response)

if __name__ == "__main__":
    main()
