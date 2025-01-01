# PDF Chatbot with Appointment Booking System

A powerful Streamlit application that combines PDF document chat capabilities with an intelligent appointment booking system using LangChain and Google's Gemini Pro model.

## 🌟 Features

- **PDF Document Processing**
  - Upload and process multiple PDF documents
  - Extract and chunk text content
  - Create searchable vector embeddings
  - Intelligent question-answering based on document content

- **Smart Appointment Booking**
  - Natural language date parsing
  - Email and phone number validation
  - Automated email confirmation
  - Appointment storage in JSON format
  - LangChain Tools integration for enhanced functionality

- **User-Friendly Interface**
  - Clean and intuitive Streamlit UI
  - Real-time processing feedback
  - Error handling and user guidance
  - Seamless integration of both chatbot and booking features

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/sobit-nep/AI-Document-QA-Appointment-Booking.git
cd pdf-chatbot-appointment
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install streamlit PyPDF2 langchain-community langchain-google-genai python-dotenv validators email-validator google-generativeai faiss-cpu
```

## ⚙️ Configuration

1. Create a `.env` file in the project root directory with the following variables:
```env
GOOGLE_API_KEY=your_google_api_key
EMAIL_ADDRESS=your_email_address
EMAIL_PASSWORD=your_email_app_password
```

Note: 
- Get your Google API key from [Google AI Studio](https://aistudio.google.com/apikey)
- For Gmail, use an App Password instead of your regular password. [Learn how to create one](https://support.google.com/accounts/answer/185833?hl=en)

## 🚀 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Upload PDF Documents:
   - Use the sidebar to upload one or more PDF files
   - Click "Process Documents" to extract and index the content

3. Interact with the Chatbot:
   - Ask questions about the content of your uploaded PDFs
   - The system will provide relevant answers based on the document content

4. Book Appointments:
   - Type keywords like "appointment", "schedule", or "book" to trigger the booking flow
   - Fill in your details (name, email, phone, preferred date)
   - Receive email confirmation upon successful booking

## 🏗️ Project Structure

```
pdf-chatbot-appointment/
├── app.py                 # Main application file
├── .env                   # Environment variables
├── appointments.json      # Stored appointments
├── faiss_index/          # Vector store for PDF content
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## 💡 Key Components

- **DocumentProcessor**: Handles PDF processing and text extraction
- **ChatTools**: Manages validation, date parsing, and email functions
- **ChatbotUI**: Combines PDF chat and appointment booking functionality
- **LangChain Tools**: Provides structured agent-based interactions

## 📝 Important Notes

- Ensure your Google API key has access to the Gemini model
- Configure your email settings properly for appointment confirmations
- The system stores appointments locally in a JSON file
- Vector store is created in the local directory



## ✨ Future Improvements

- Implement user authentication
- Add calendar integration for appointment scheduling
- Create a dashboard for appointment management
- Add support for multiple languages
- Implement conversation history
