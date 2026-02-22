# Multi-PDF AI Assistant Chat App

## Introduction
------------
Multi-PDF AI Assistant is a Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF files and ask questions about their content.

The system processes PDFs, converts them into embeddings, stores them in a vector database (FAISS), and uses a Large Language Model (LLM) to generate answers based on the uploaded documents.

**Built with:**
- Streamlit (Frontend UI)
- LangChain (RAG Pipeline)
- FAISS (Vector Database)
- HuggingFace Embeddings
- Groq LLM API

## How It Works
------------

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. Upload PDFs: Users upload one or multiple PDF files through the web interface.

2. Text Extraction & Embedding: 

- Extracts text from PDFs
- Cleans and merges text
- Splits text into smaller chunks for better retrieval
- Text chunks are converted into vector embeddings using: intfloat/e5-base-v2


4. Question Answering: When you ask a question, 
- The system converts the question into embeddings.
- It searches for the most relevant document chunks.
- Relevant context is passed to the LLM.
- The LLM generates an answer based only on retrieved content.

This is called:

> **Retrieval-Augmented Generation (RAG)**

## Dependencies and Installation
----------------------------
To install the Multi-PDF AI Assistant Chat App, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from Qroq (or openai) | Huggingface | Langsmith (optional) and add it to the `.env` file in the project directory.
```commandline
GROQ_API_KEY=your_groq_api_key
```

## Usage
-----
To use the Multi-PDF AI Assistant Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.

## Author
------------
Built as a personal AI project to learn RAG systems and production-level AI application architecture.

@linkedin : [text](https://www.linkedin.com/in/abdallahsabry1/)
