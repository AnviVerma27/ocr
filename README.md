# **Document OCR and Search Application**

This project is a web-based application that allows users to upload documents (in PDF or image formats), extract text from them using Optical Character Recognition (OCR), and perform keyword searches within the extracted text. The application leverages state-of-the-art AI models for text extraction and supports multilingual document processing, specifically handling both English and Hindi text. Additionally, it provides a search functionality that highlights occurrences of user-specified keywords in the extracted content.

## **Setup Instructions**

### **1. Prerequisites**
   - **Python 3.8+**: Make sure you have Python installed on your system.
   - **CUDA** (Optional): For GPU support, ensure CUDA is installed and available.
   - **Streamlit**: To install Streamlit for running the web application.

### **2. Installation Steps**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AnviVerma27/ocr.git
   cd ocr-search-app
2. **Install Dependencies: Create and activate a virtual environment (optional but recommended)**:
   ```bash
    python -m venv venv
    source venv/bin/activate   # For Linux/MacOS
    venv\Scripts\activate      # For Windows
    pip install -r requirements.txt
3. **Run the application**:
   ```bash
   streamlit run test2.py

## **Deployment on Hugging Face Spaces**

This application is deployed on [Hugging Face Spaces](https://huggingface.co/spaces/anvi27/ocr) using Streamlit. You can interact with the application directly in your browser without needing to install any dependencies or set up a local environment.

## **Features**

1. **Upload PDF or Image Documents**: Users can upload document in image formats (JPG, PNG, JPEG).
   
2. **Text Extraction from Documents**: 
   - The application supports OCR for both English and Hindi text from the documents.

3. **View Extracted Text**: Once extracted, the text is displayed in JSON format for clarity and further usage.

4. **Keyword Search within Extracted Text**: 
   - Users can search for specific keywords in the extracted text.
   - The search results return the entire line containing the keyword, with the keyword highlighted.

5. **Support for Advanced AI Models**: 
   - Utilizes the `Qwen2-VL-7B-Instruct` model for advanced visual and language understanding.
   - Uses `RAGMultiModalModel` for indexing and efficient querying of document content.

## **Technologies Used**

- **Python**: The core language for developing this application.
- **Streamlit**: Used for building the user interface and handling file uploads, display, and form input.
- **byaldi**: Handles multimodal information and indexing using RAG (Retrieval-Augmented Generation) models.
- **transformers**: From Hugging Face, specifically used for handling and processing the `Qwen2-VL` model.
- **Regular Expressions (re)**: Used to perform keyword searches within the extracted text.

## **How It Works**

### 1. **File Upload**
   - The user uploads a document through the Streamlit interface. The document can be a PNG, JPG, or JPEG.
     
### 2. **Text Extraction**
   - The uploaded document is processed using the `Qwen2-VL` model. This model is capable of handling both text and visual inputs, making it well-suited for OCR tasks.
   - The document image is indexed using the `RAGMultiModalModel` for efficient querying.
   - A query is sent to extract all English and Hindi text from the document.
   - The extracted text is stored in the session state to persist across reruns, ensuring that text is only extracted once for a given document.

### 3. **Displaying Extracted Text**
   - The extracted text is displayed in JSON format, making it easy for users to review and understand.

### 4. **Search Functionality**
   - Users can input a keyword to search for within the extracted text.
   - The application splits the extracted text into individual lines and searches for the keyword in each line.
   - If the keyword is found in a line, the entire line is returned with the keyword highlighted.
   - If no matches are found, a message is displayed indicating that the keyword was not found.

