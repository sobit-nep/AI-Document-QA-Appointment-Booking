import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import datetime
import re
import validators
import json
from pathlib import Path

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key is missing. Please configure it in your .env file.")
    st.stop()

# Configure Google Generative AI
from google.generativeai import configure
configure(api_key=api_key)

# PDF Handling
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

# Text Splitting
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Vector Store
def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")

# Conversational Chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, say: "Answer is not available in the context." Do not guess answers.
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# User Query Handling
def user_query(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response.get("output_text", "Error: No response generated.")
    except Exception as e:
        return f"Error handling query: {e}"

# Validate User Input
def validate_input(input_type, value):
    if input_type == "email":
        value = value.strip()
        return bool(validators.email(value))
    elif input_type == "phone":
        value = value.strip().replace(" ", "")
        return bool(re.match(r'^\+?\d{9,15}$', value))
    return False

# Extract Date Using LLM
def extract_date_llm(date_query):
    try:
        # Define the prompt to compute the relative date
        prompt = PromptTemplate(
            template="""
            Today's date is {current_date}.
            Compute the exact date for the following query:
            "{date_query}"
            Provide the result in YYYY-MM-DD format.
            """,
            input_variables=["current_date", "date_query"]
        )
        # Set up the LLM
        llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Current date as context
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Query the LLM
        response = chain.run({"current_date": current_date, "date_query": date_query})
        return response.strip()  # Ensure the response is clean
    except Exception as e:
        return f"Error parsing date with LLM: {e}"

# Save Appointment Info
def save_appointment(name, phone, email, date):
    try:
        data = {
            "name": name,
            "phone": phone,
            "email": email,
            "appointment_date": date
        }
        
        file_path = Path("appointments.json")
        
        if file_path.exists():
            with open(file_path, "r") as file:
                appointments = json.load(file)
        else:
            appointments = []

        appointments.append(data)

        with open(file_path, "w") as file:
            json.dump(appointments, file, indent=4)
        
        return True
    except Exception as e:
        st.error(f"Error saving appointment: {e}")
        return False

# Main Application
def main():
    st.set_page_config(page_title="AI Chatbot with Appointment Booking", layout="wide")
    st.header("Document Chatbot with Appointment Booking 💬📅")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Documents processed!")
            else:
                st.error("Please upload PDF documents.")

    user_question = st.text_input("Ask a Question:")
    if user_question:
        if "call me" in user_question.lower() or "appointment" in user_question.lower():
            st.write("Let's collect some details for a callback or appointment!")
            name = st.text_input("Name:")
            phone = st.text_input("Phone Number:")
            email = st.text_input("Email:")
            date_query = st.text_input("When should we contact you? (e.g., 'next Monday')")

            if st.button("Submit Details"):
                if validate_input("email", email) and validate_input("phone", phone):
                    date = extract_date_llm(date_query)
                    if "Error" not in date:
                        if save_appointment(name, phone, email, date):
                            st.success(f"Appointment booked! We'll contact you on {date}.")
                    else:
                        st.error("Invalid date format. Please try again.")
                else:
                    st.error("Invalid email or phone number. Please try again.")
        else:
            with st.spinner("Processing your question..."):
                response = user_query(user_question)
                st.write(response)

if __name__ == "__main__":
    main()
