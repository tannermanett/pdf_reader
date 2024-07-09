import os
import logging
import PyPDF2
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader

# Set up logging
logging.basicConfig(level=logging.DEBUG)

def get_index(data, index_name, rebuild=False):
    index_path = os.path.join("indices", index_name)
    if not os.path.exists(index_path) or rebuild:
        if rebuild:
            logging.debug(f"Rebuilding index: {index_name}")
            if os.path.exists(index_path):
                for root, dirs, files in os.walk(index_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                logging.debug("Existing index removed.")
        else:
            logging.debug(f"Building new index: {index_name}")

        os.makedirs(index_path, exist_ok=True)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_path)
    else:
        logging.debug(f"Loading index from storage: {index_name}")
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_path))
    return index

def parse_pdf(file_path):
    logging.debug(f"Parsing PDF file: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")
    
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text
            logging.debug(f"Extracted text from page {page_num}: {page_text}")
    
    logging.debug(f"Total extracted text length: {len(text)}")
    return text

def load_documents_and_build_index(pdf_file_path, index_name, rebuild=False):
    text = parse_pdf(pdf_file_path)
    if not text:
        raise ValueError("No text found in the PDF file.")
    
    logging.debug("Loading document...")
    reader = SimpleDirectoryReader(input_files=[pdf_file_path])
    data = reader.load_data()
    logging.debug("Document loaded.")

    index = get_index(data, index_name, rebuild=rebuild)
    return index.as_query_engine()

