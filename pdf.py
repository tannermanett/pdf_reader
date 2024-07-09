import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage, SimpleDirectoryReader
from dotenv import load_dotenv
import PyPDF2

# Load environment settings
load_dotenv()

def get_index(data, index_name, rebuild=False):
    """
    Build or load an index based on the provided data and index name.
    Adds support for forced rebuild.
    """
    index_path = os.path.join("indices", index_name)  # Storing index in a subdirectory for organization
    if not os.path.exists(index_path) or rebuild:
        if rebuild:
            print("Rebuilding index:", index_name)
            # Remove existing index files if rebuilding
            if os.path.exists(index_path):
                for root, dirs, files in os.walk(index_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                print("Existing index removed.")
        else:
            print("Building new index:", index_name)

        os.makedirs(index_path, exist_ok=True)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_path)
    else:
        print("Loading index from storage:", index_name)
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_path)
        )
    return index

def parse_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def load_documents_and_build_index(pdf_file, index_name, rebuild=False):
    """
    Load a PDF document from a file, build or reload an index, and return a query engine.
    """
    text = parse_pdf(pdf_file)
    
    if not text:
        raise ValueError("No text found in the PDF file.")
    
    print("Loading document...")
    reader = SimpleDirectoryReader(input_files=[pdf_file])
    data = reader.load_data()
    print("Document loaded.")

    # Build or load the index, with optional rebuild
    index = get_index(data, index_name, rebuild=rebuild)
    return index.as_query_engine()
