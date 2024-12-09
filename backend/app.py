import os
import traceback
import numpy as np
import json
from flask import Flask, request, jsonify
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from boto3.dynamodb.conditions import Attr
from flask_cors import CORS  
from pypdf import PdfReader
import os
import re
import json
import boto3
import requests
from openai import OpenAI
from google.auth.transport.requests import Request
from google.oauth2.id_token import fetch_id_token



# Ensure fallback for unsupported operations on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
PINECONE_KEY = os.getenv("PINECONE_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION")
TABLE_NAME = os.getenv("TABLE_NAME")
TOKEN = os.getenv("TOKEN")
LLAMA_URL = os.getenv("LLAMA_URL")
BACKEND_DOMAIN = os.getenv("BACKEND_DOMAIN")


# Initialize DynamoDB Resource
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)


pc = Pinecone(api_key=PINECONE_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

openAIClient = OpenAI(api_key=OPENAI_API_KEY,)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
    
# Split text into manageable chunks
def split_text_with_langchain(text, chunk_size=4096, chunk_overlap=200):
    """
    Splits the text into manageable chunks using LangChain's RecursiveCharacterTextSplitter.
    Args:
        text (str): The text to split.
        chunk_size (int): Maximum size of each chunk in tokens.
        chunk_overlap (int): Number of overlapping characters between chunks.
    Returns:
        List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def parse_pdf_to_text(pdf_file):
    """
    Extracts text from an uploaded PDF file.
    Args:
        pdf_file (FileStorage): The uploaded PDF file from the request.
    Returns:
        str: Extracted text from the PDF.
    """
    reader = PdfReader(pdf_file)
    number_of_pages = len(reader.pages)
    text = ""
    for i in range(number_of_pages):
        page = reader.pages[i]
        text += page.extract_text()
    return text

def clean_filename(filename):
    """
    Cleans special characters from the filename to make it filesystem-safe.
    Args:
        filename (str): Original filename.
    Returns:
        str: Cleaned filename.
    """
    return re.sub(r'[^A-Za-z0-9]', '_', os.path.splitext(filename)[0])

def save_text_to_temp_file(text, folder_path, original_filename):
    """
    Saves the extracted text to a temporary file in a given folder.
    Args:
        text (str): The extracted text from the PDF.
        folder_path (str): Path to the folder where the text file will be saved.
        original_filename (str): The original filename of the PDF file.
    Returns:
        str: Path to the saved text file.
    """
    os.makedirs(folder_path, exist_ok=True)
    cleaned_filename = clean_filename(original_filename)
    text_file_path = os.path.join(folder_path, f"{cleaned_filename}.txt")
    with open(text_file_path, "w", encoding="utf-8") as file:
        file.write(text)
    return text_file_path

def get_embedding(text, model="text-embedding-3-small"):
    embeddings = []
    for chunk in text: 
        try:
            chunk = chunk.page_content.replace("\n", " ")
            embeddings.append(openAIClient.embeddings.create(input = [chunk], model=model).data[0].embedding)
        except Exception as e:
            chunk = chunk.replace("\n", " ")
            embeddings.append(openAIClient.embeddings.create(input = [chunk], model=model).data[0].embedding)
    return embeddings

def query_pinecone(query, top_k=10):
    query_embedding = get_embedding([query])
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return results

def get_auth_token():
    auth_request = Request()
    target_audience = LLAMA_URL
    id_token = fetch_id_token(auth_request, target_audience)
    return id_token

def generate_answer(query, matches):
    context = " ".join(
        [match.get("metadata", {}).get("chunk", "") for match in matches if "metadata" in match]
    )
    input_text = f"This is my question: {query}, please answer the question based on the following context: {context}\n\nDo not mention that you have been given context\n"

    print(input_text)

    auth_token = get_auth_token()

    headers = {
    "Authorization": f"Bearer {auth_token}", 
    "Content-Type": "application/json"}
    data = {
        "model": "llama3.2:3b",
        "prompt": input_text,
        "stream": False}
    print("generating answer")
    try:
        response = requests.post(LLAMA_URL, json=data, headers=headers)
    except Exception as e:
        print(f"Error: {e}")
    print("generation done")
    return response

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query_text = data['query']
        print(f"1: query text: {query_text}")
        pinecone_results = query_pinecone(query_text)
        matches = pinecone_results.get("matches", [])
        if not matches:
            return jsonify({"answer": "No relevant matches found in the database."})
        
        paper_ids = list(set(int(match['id'].split("#")[0].split("_")[1]) for match in matches))[:5]
        dynamo_response = getPapersFromDynamo(paper_ids)
        print(f"2: Paper IDs: {paper_ids}")
        print(f"4: Dynamo Response :{dynamo_response}")

        if type(dynamo_response) == str:
            return jsonify({"error": "Failed to retrieve data from DynamoDB.", "details": dynamo_response}), 500
        
        answer = generate_answer(query_text, matches[:3])
        return jsonify({"result": str(matches), "dynamo_data": dynamo_response, "answer": answer.text})
    except Exception as e:
        print("exception raised")
        error_trace = traceback.format_exc()
        print(f"Error: {error_trace}")
        return jsonify({"error": str(e), "trace": error_trace}), 500


@app.route('/summarize', methods=['POST'])
def summarize():    

    """
    Summarizes the uploaded PDF file.
    Expects a PDF file to be uploaded as a POST request.
    """
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        original_filename = file.filename

        # # Extract text from the PDF
        text = parse_pdf_to_text(file)

        # Optionally save the extracted text to a temporary file
        temp_folder = "/tmp/pdf_texts"  # Define a temporary folder for saving text files
        temp_file_path = save_text_to_temp_file(text, temp_folder, original_filename)

        query = "Summarize the content clearly and concisely with a maximum word limit of 300 words."

        input_text = f"Query: {query}\nContext: {text}\n\nProvide a detailed summary based on the context.\n"

        headers = {
        "Authorization": "Bearer "+ TOKEN, 
        "Content-Type": "application/json"}
        data = {
            "model": "llama3.2:3b",
            "prompt": input_text,
            "stream": False}
        print("generating answer")
        try:
            # print("test")
            response = requests.post(LLAMA_URL, json=data, headers=headers)
        except Exception as e:
            print(e)

        print("generation done")
        return jsonify({"message": str("response.json()")}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/addToDynamo', methods=['POST'])
def addToDynamo():
    try:
        with open('pdf_metadata.json', 'r') as file:
            pdf_metadata = json.load(file)

        table = dynamodb.Table(TABLE_NAME)

        for cleaned_filename, metadata in pdf_metadata.items():
            data_to_add = {
                "PaperTxtName": cleaned_filename + ".txt", 
                "PaperID": metadata[0],  
                "PaperLink": metadata[1],
                "PaperPDFName": metadata[2]  
            }

            table.put_item(Item=data_to_add)

        return jsonify({"message": "All items added successfully!"}), 200


    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/getFromDynamo', methods=['POST'])
def getFromDynamo():
    try:
        request_data = request.get_json()
        paper_ids = request_data.get("PaperIDs")
        results = getPapersFromDynamo(paper_ids)

        if type(results) == str:
            return jsonify({"error": "Failed to retrieve data from DynamoDB.", "details": results}), 500

        return jsonify({"data": results}), 200

    except Exception as e:
        print(f"2.5: Exception: {e}")
        return jsonify({"error": str(e)}), 500

def getPapersFromDynamo(paper_ids):
    try:
        print(f"2.1: Paper IDs: {paper_ids}")

        if not paper_ids or not isinstance(paper_ids, list):
            print(f"2.2: If statement")
            return "Invalid input. Please provide a list of PaperIDs."

        table = dynamodb.Table(TABLE_NAME)
        print(f"2.3: Table name: {table}")

        results = []
        for paper_id in paper_ids:
            response = table.scan(
                FilterExpression=Attr("PaperID").eq(paper_id)
            )

            if 'Items' in response and response['Items']:
                results.extend(response['Items'])
        print(f"2.4: Results: {results}")

        return results

    except Exception as e:
        print(f"2.5: Exception: {e}")
        return str(e)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050, debug=True)