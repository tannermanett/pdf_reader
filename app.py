from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pdf import parse_pdf, load_documents_and_build_index

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    
    if not query or not index_name:
        return jsonify({"error": "Invalid request"}), 400
    
    # Load the index and build the query engine
    pdf_file = data.get('pdf_file_path')
    rebuild = data.get('rebuild', False)
    engine = load_documents_and_build_index(pdf_file, index_name, rebuild=rebuild)
    
    response = engine.query(query)
    
    # Convert the response to a JSON-serializable format
    serialized_response = {
        "query": query,
        "response": str(response)  # Convert the Response object to a string
    }

    return jsonify(serialized_response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
