import os
import traceback
import numpy as np
import random
import json
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from pinecone import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from boto3.dynamodb.conditions import Attr
import torch
from flask_cors import CORS  
from pypdf import PdfReader
import os
import re
import json
import boto3
import requests
from openai import OpenAI
from together import Together
import vertexai
from vertexai.generative_models import GenerativeModel


# Ensure fallback for unsupported operations on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
PINECONE_KEY = "pcsk_733LpQ_3vUPKxHKBvL21VL4JaZVMoW1XxF2xFyWFZDx5brAgnbrQ5UVoySp7ad26QU416D" #Updated to access the openAI embedding
PINECONE_INDEX_NAME = "v2-research-paper-index" #Pinecone index name
OPENAI_API_KEY="sk-cZPrWqOChvs4ZVzmnN86oimk0uame9WaV07CVLKIWzT3BlbkFJdoBT2V24zTFu_mUzWOnwNqv-qkgnQPtcB3ErD3vw8A" #OpenAI api key
AWS_ACCESS_KEY_ID = "AKIA6ODU6VDBSWRFQJ6F"  # Replace with your Access Key
AWS_SECRET_ACCESS_KEY = "gTdXAvAkOcAhBU6UIbQWehkv1L/N/WtNB/4MoPgW"  # Replace with your Secret Key
AWS_REGION = "us-west-2"  # Replace with your AWS Region
TABLE_NAME = "pdf_metadata"  # Replace with your DynamoDB table name
TOKEN  = "eyJhbGciOiJSUzI1NiIsImtpZCI6IjJjOGEyMGFmN2ZjOThmOTdmNDRiMTQyYjRkNWQwODg0ZWIwOTM3YzQiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJhY2NvdW50cy5nb29nbGUuY29tIiwiYXpwIjoiNjE4MTA0NzA4MDU0LTlyOXMxYzRhbGczNmVybGl1Y2hvOXQ1Mm4zMm42ZGdxLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwiYXVkIjoiNjE4MTA0NzA4MDU0LTlyOXMxYzRhbGczNmVybGl1Y2hvOXQ1Mm4zMm42ZGdxLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwic3ViIjoiMTEwNjE4MjQ3NDgxMzA0MTY2OTAwIiwiaGQiOiJueXUuZWR1IiwiZW1haWwiOiJ2ZzI1MjNAbnl1LmVkdSIsImVtYWlsX3ZlcmlmaWVkIjp0cnVlLCJhdF9oYXNoIjoiaGw1eTBac3lIdTVzUGJlYWUxaW1kZyIsIm5iZiI6MTczMzc3MzU5NiwiaWF0IjoxNzMzNzczODk2LCJleHAiOjE3MzM3Nzc0OTYsImp0aSI6IjQ5NTBlOWZmYmRjMTUwYTE4Yzc1MzczZjdiYzIyZGMxOTA1NjAyOWUifQ.VrxAwriqGyBEczyXYu2f8bJMaXWg3KhmdD9GXzTAbAIhegRylZx_QNvDVVHsS0CJ6uA19hzkPUz0dldXm4MazHEYOp92faEBS44lOuc0_GevkEwdmGPE1HtTxcWFEtUiHoUzYzB41M1yPZF_JTg0EUJ2KhFMt-1m2dzGW1PbrusHL60lNg-1xqrWj8DU4R9W0hs0MPMI1doYxy8TFqPHeFVi3DcbzlqFO81lrO8wHXvrVBeYvlsgi57rhMobyEO9NLYS6O8R_qflEfVDQqhcIalCU-rRXz7KPwtX5ycMN4MQI_-vJTjNlDcHt7LqGYaRtInsvv0go7Nbm6meoAms1Q"
LLAMA_URL = "https://ollama-llama32-316797979759.us-east4.run.app/api/generate"
TOGETHER_API_KEY = "7da8c3e879a9b326564bca00116aaf5845eabd3cd360ad38c18c8bcfec7c3b2d"


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

client = Together(api_key=TOGETHER_API_KEY)
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

def generate_answer(query, matches, model_type):
    context = " ".join(
        [match.get("metadata", {}).get("chunk", "") for match in matches if "metadata" in match]
    )
    input_text = f"Query: {query}\nContext: {context}\n\nBased on the Context please answer the Query\n"
    if model_type == "llama3.2":
        headers = {
        "Authorization": "Bearer "+ TOKEN, 
        "Content-Type": "application/json"}
        data = {
            "model": "llama3.2:3b",
            "prompt": input_text,
            "stream": False}
        print("generating answer")
        responseAPI = requests.post(LLAMA_URL, json=data, headers=headers)
        response = responseAPI.text
        print("generation done")
    elif model_type == "llama3.3":
        responseAPI = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
                {"role": "system", "content": "You are a helpful assistant for the conducting research that answers questions based on provided context. Don't mention anything about the context when answering"},
                {"role": "user", "content": input_text}
            ],
        stream=False)
        print("generating answer")
        response = responseAPI.choices[0].message.content
        print("generation done")
    elif model_type == "gemini1.5":
        PROJECT_ID = "lofty-cabinet-443918-a9"
        vertexai.init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel("gemini-1.5-flash-002")
        print("generating answer")
        responseAPI = model.generate_content(input_text)
        response = responseAPI.candidates[0].content.parts[0].text
        print("generation done")
    return response

@app.route('/query', methods=['POST'])
def query():
    try:
        data = request.json
        query_text = data['query']
        model_type = data['model'] 
        pinecone_results = query_pinecone(query_text)
        matches = pinecone_results.get("matches", [])
        if not matches:
            return jsonify({"answer": "No relevant matches found in the database."})
        
        paper_ids = list(set(int(match['id'].split("#")[0].split("_")[1]) for match in matches))[:5]
        dynamo_response = requests.post(
            "http://127.0.0.1:5000/getFromDynamo",  # Replace with the actual URL if hosted elsewhere
            json={"PaperIDs": paper_ids}
        )

        if dynamo_response.status_code != 200:
            return jsonify({"error": "Failed to retrieve data from DynamoDB.", "details": dynamo_response.json()}), 500
        
        dynamo_data = dynamo_response.json()
        answer = generate_answer(query_text, matches[:5], model_type)
        return jsonify({"result": str(matches), "dynamo_data": dynamo_data, "answer": answer})
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

        if not paper_ids or not isinstance(paper_ids, list):
            return jsonify({"error": "Invalid input. Please provide a list of PaperIDs."}), 400

        table = dynamodb.Table(TABLE_NAME)

        results = []
        for paper_id in paper_ids:
            response = table.scan(
                FilterExpression=Attr("PaperID").eq(paper_id)
            )

            if 'Items' in response and response['Items']:
                results.extend(response['Items'])

        return jsonify({"data": results}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)