# AI Document QA with Appointment Booking

## Overview

This project combines the power of AI to create a chatbot capable of answering questions from uploaded documents and scheduling appointments. It uses **Google Generative AI** for natural language understanding, making it highly effective for document-based question answering and relative date parsing for appointment scheduling.

## Features

- **Document-Based Question Answering**:
  - Upload PDF documents and ask questions directly related to the content.
  - Uses AI-powered embeddings for context-aware responses.

- **Appointment Booking**:
  - Schedule appointments using natural language queries like "next Monday" or "three weeks from now."
  - Automatically parses and validates user input for phone numbers, email, and appointment dates.

- **Streamlined AI Integration**:
  - Powered by **Google Generative AI** for both chat and embedding functionalities.

## Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) for the user interface.
- **Backend**: Python with libraries such as:
  - `langchain` for AI chains.
  - `PyPDF2` for PDF handling.
  - `FAISS` for vector storage.
  - `dateparser` for fallback date parsing.
  - `validators` for input validation.
- **AI Integration**: Google Generative AI via the `langchain_google_genai` library.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/ai-document-qa-appointment-booking.git
   cd ai-document-qa-appointment-booking
   ```

2. **Set Up Virtual Environment** (Optional but Recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory with the following content:
     ```dotenv
     GOOGLE_API_KEY=your_google_api_key
     ```

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Document QA**:
   - Upload PDF documents using the sidebar.
   - Once processed, you can ask questions about the content.

2. **Appointment Booking**:
   - Ask for an appointment using natural language, e.g., "Schedule a meeting next Tuesday."
   - Fill in your name, email, phone number, and preferred date.

3. **View Results**:
   - The application will display answers to your questions and confirmation of your scheduled appointments.

## Repository Structure

```
ai-document-qa-appointment-booking/
│
├── app.py                   # Main application script
├── requirements.txt         # Python dependencies
├── appointments.json        # Saved appointment data
├── .env                     # Environment variables (not included in repo)
├── README.md                # Project documentation
```





## Acknowledgments

- [Google Generative AI](https://ai.google/) for powering the chatbot.
- [LangChain](https://langchain.com/) for streamlining AI chain integrations.
- [Streamlit](https://streamlit.io/) for providing an easy-to-use framework for building AI-powered web apps.

---


### What’s Included in the `README.md`:
1. **Overview and Features**: Highlights what the application does.
2. **Tech Stack**: Details the technologies used.
3. **Installation Instructions**: Steps to get the app running locally.
4. **Usage Instructions**: How users can interact with the app.
5. **Repository Structure**: Directory layout for better organization.
