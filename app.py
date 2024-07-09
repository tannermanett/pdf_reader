from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
from pdf import parse_pdf, load_documents_and_build_index

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/parse_pdf', methods=['POST'])
def parse_pdf_endpoint():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    text = parse_pdf(file)
    return jsonify({"text": text})

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('query')
    index_name = data.get('index_name')
    pdf_file_path = data.get('pdf_file_path')
    rebuild = data.get('rebuild', False)
    
    logging.debug(f"Received request with query: {query}")
    logging.debug(f"Index name: {index_name}, PDF file path: {pdf_file_path}, Rebuild: {rebuild}")
    
    if not query or not index_name or not pdf_file_path:
        return jsonify({"error": "Invalid request"}), 400
    
    try:
        engine = load_documents_and_build_index(pdf_file_path, index_name, rebuild=rebuild)
        response = engine.query(query)
        
        serialized_response = {
            "query": query,
            "response": str(response)
        }

        return jsonify(serialized_response)
    except FileNotFoundError as fnf_error:
        logging.error(f"File not found: {fnf_error}")
        return jsonify({"error": str(fnf_error)}), 404
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
