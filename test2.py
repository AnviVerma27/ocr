import streamlit as st
from PIL import Image
from pdf2image import convert_from_path
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import time
import json
import re

# Check device availability (GPU/CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to load models only once
@st.cache_resource
def initialize_models():
    # Load models for text extraction
    multimodal_model = RAGMultiModalModel.from_pretrained("vidore/colpali")
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)

    return multimodal_model, qwen_model, qwen_processor

multimodal_model, qwen_model, qwen_processor = initialize_models()

# Upload section
st.title("Document Text Extraction")
doc_file = st.file_uploader("Upload Image File", type=[ "png", "jpg", "jpeg"])

# Store extracted text across reruns
if "document_text" not in st.session_state:
    st.session_state.document_text = None

if doc_file is not None:
    # Check file extension
    file_ext = doc_file.name.split('.')[-1].lower()
    document_image = Image.open(doc_file)  # Handle image files directly

    # Display uploaded document image
    st.image(document_image, caption="Document Preview", use_column_width=True)

    # Create a unique index name for the document
    index_id = f"doc_index_{int(time.time())}"  # Timestamp-based unique index

    # Only process if text hasn't been extracted yet
    if st.session_state.document_text is None:
        st.write(f"Indexing document with unique ID: {index_id}...")
        temp_image_path = "temp_image.png"
        document_image.save(temp_image_path)
        
        # Index the image using multimodal model
        multimodal_model.index(
            input_path=temp_image_path,
            index_name=index_id,
            store_collection_with_index=False,
            overwrite=False
        )

        # Define the extraction query
        extraction_query = "Extract all English and Hindi text from this document"
        st.write("Querying the document with the extraction query...")

        # Search results from RAG
        search_results = multimodal_model.search(extraction_query, k=1)

        # Prepare input data for Qwen model
        input_message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": document_image},
                    {"type": "text", "text": extraction_query},
                ],
            }
        ]

        # Prepare inputs for Qwen2-VL
        input_text = qwen_processor.apply_chat_template(input_message, tokenize=False, add_generation_prompt=True)
        vision_inputs, _ = process_vision_info(input_message)

        model_inputs = qwen_processor(
            text=[input_text],
            images=vision_inputs,
            padding=True,
            return_tensors="pt",
        )

        model_inputs = model_inputs.to(device)

        # Generate text output from the image using Qwen2-VL model
        st.write("Generating extracted text...")
        output_ids = qwen_model.generate(**model_inputs, max_new_tokens=100)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, output_ids)
        ]

        extracted_output = qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Store the extracted text in session state
        st.session_state.document_text = extracted_output[0]

    # Display extracted text in JSON format
    extracted_text_content = st.session_state.document_text
    structured_data = {"extracted_text": extracted_text_content}

    st.subheader("Extracted Text in JSON:")
    st.json(structured_data)

# Implement search functionality in extracted text
if st.session_state.document_text:
    with st.form(key='text_search_form'):
        search_input = st.text_input("Enter a keyword to search within the extracted text:")
        search_action = st.form_submit_button("Search")

    if search_action and search_input:
        # Split the extracted text into lines for searching
        full_text = st.session_state.document_text
        lines = full_text.split('\n')

        results = []
        # Search for keyword in each line and collect lines that contain the keyword
        for line in lines:
            if re.search(re.escape(search_input), line, re.IGNORECASE):
                # Highlight keyword in the line
                highlighted_line = re.sub(f"({re.escape(search_input)})", r"**\1**", line, flags=re.IGNORECASE)
                results.append(highlighted_line)

        # Display search results
        st.subheader("Search Results:")
        if not results:
            st.write("No matches found.")
        else:
            for result in results:
                st.markdown(result)
