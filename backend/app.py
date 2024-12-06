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
import boto3

# Ensure fallback for unsupported operations on MPS
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Pinecone credentials
api_key = "pcsk_bYpHQ_MYJaBXuyz9jvAKrVDCJ9GDWQAS2cPFufcQmgJN8UE6oVzYrMYg3tp4cJ1RV4nVb"
index_name = "v2-research-paper-index"

AWS_ACCESS_KEY_ID = "AKIA6ODU6VDBSWRFQJ6F"  # Replace with your Access Key
AWS_SECRET_ACCESS_KEY = "gTdXAvAkOcAhBU6UIbQWehkv1L/N/WtNB/4MoPgW"  # Replace with your Secret Key
AWS_REGION = "us-west-2"  # Replace with your AWS Region
TABLE_NAME = "pdf_metadata"  # Replace with your DynamoDB table name

# Initialize DynamoDB Resource
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)


pc = Pinecone(api_key=api_key)
index = pc.Index(index_name)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# # Load SentenceTransformer for embeddings
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# # embedding_model = SentenceTransformer('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True).cuda()  # Lightweight model for sentence embeddings

# # Load LLaMA model
# hf_token = "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"
# llama_model_name = "meta-llama/Llama-2-7b-chat-hf"

# device = "cuda" if torch.cuda.is_available() else "cpu"

# llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=hf_token)
# llama_model = AutoModelForCausalLM.from_pretrained(
#     llama_model_name,
#     device_map="auto" if torch.cuda.is_available() else None,
#     torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#     use_auth_token=hf_token
# )

# # Add padding token if necessary
# if llama_tokenizer.pad_token is None:
#     llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#     llama_model.resize_token_embeddings(len(llama_tokenizer))


# # Load the summarizer model
# def load_summarizer(model_name="t5-small"):
#     """
#     Load the summarization model pipeline.
#     Args:
#         model_name (str): The name of the Hugging Face model.
#     Returns:
#         summarizer function
#     """
#     if model_name.startswith("t5"):
#         # Use T5 summarizer
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#         tokenizer = AutoTokenizer.from_pretrained(model_name)

#         def t5_summarizer(text):
#             input_ids = tokenizer.encode(f"summarize: {text}", return_tensors="pt", truncation=True, max_length=512)
#             outputs = model.generate(input_ids, max_length=130, min_length=30, length_penalty=2.0, num_beams=4)
#             return tokenizer.decode(outputs[0], skip_special_tokens=True)
        
#         return t5_summarizer
#     else:
#         return None
    
# # Split text into manageable chunks
# def split_text_with_langchain(text, chunk_size=4096, chunk_overlap=200):
#     """
#     Splits the text into manageable chunks using LangChain's RecursiveCharacterTextSplitter.
#     Args:
#         text (str): The text to split.
#         chunk_size (int): Maximum size of each chunk in tokens.
#         chunk_overlap (int): Number of overlapping characters between chunks.
#     Returns:
#         List of text chunks.
#     """
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def parse_pdf_to_text(pdf_file):
#     """
#     Extracts text from an uploaded PDF file.
#     Args:
#         pdf_file (FileStorage): The uploaded PDF file from the request.
#     Returns:
#         str: Extracted text from the PDF.
#     """
#     reader = PdfReader(pdf_file)
#     number_of_pages = len(reader.pages)
#     text = ""
#     for i in range(number_of_pages):
#         page = reader.pages[i]
#         text += page.extract_text()
#     return text

# def clean_filename(filename):
#     """
#     Cleans special characters from the filename to make it filesystem-safe.
#     Args:
#         filename (str): Original filename.
#     Returns:
#         str: Cleaned filename.
#     """
#     return re.sub(r'[^A-Za-z0-9]', '_', os.path.splitext(filename)[0])

# def save_text_to_temp_file(text, folder_path, original_filename):
#     """
#     Saves the extracted text to a temporary file in a given folder.
#     Args:
#         text (str): The extracted text from the PDF.
#         folder_path (str): Path to the folder where the text file will be saved.
#         original_filename (str): The original filename of the PDF file.
#     Returns:
#         str: Path to the saved text file.
#     """
#     os.makedirs(folder_path, exist_ok=True)
#     cleaned_filename = clean_filename(original_filename)
#     text_file_path = os.path.join(folder_path, f"{cleaned_filename}.txt")
#     with open(text_file_path, "w", encoding="utf-8") as file:
#         file.write(text)
#     return text_file_path

# # Query Pinecone index
# def query_pinecone(query, top_k=5):
#     query_embedding = embedding_model.encode(query).tolist()
#     results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
#     return results

# # Generate answer using LLaMA
# def generate_answer(query, matches):
#     context = " ".join(
#         [match.get("metadata", {}).get("chunk", "") for match in matches if "metadata" in match]
#     )
#     input_text = f"Query: {query}\nContext: {context}\nAnswer:"
    
#     inputs = llama_tokenizer(
#         input_text,
#         return_tensors="pt",
#         padding=True,
#         truncation=True
#     ).to(device)
    
#     outputs = llama_model.generate(
#         inputs.input_ids,
#         max_length=300,
#         pad_token_id=llama_tokenizer.pad_token_id
#     )
#     return llama_tokenizer.decode(outputs[0], skip_special_tokens=True)

# @app.route('/')
# def home():
#     return "Welcome to the AI Query API! Use the `/query` endpoint to interact."

# @app.route('/favicon.ico')
# def favicon():
#     return '', 204

# @app.route('/query', methods=['POST'])
# def query():
#     try:
#         data = request.json
#         query_text = data['query']
        
#         pinecone_results = query_pinecone(query_text)
#         matches = pinecone_results.get("matches", [])
        
#         if not matches:
#             return jsonify({"answer": "No relevant matches found in the database."})
        
#         # answer = generate_answer(query_text, matches)
#         answer = "THIS IS MY ANSWER"
#         return jsonify({"result": str(matches)})
#     except Exception as e:
#         error_trace = traceback.format_exc()
#         print(f"Error: {error_trace}")
#         return jsonify({"error": str(e), "trace": error_trace}), 500


# @app.route('/summarize', methods=['POST'])
# def summarize():
#     """
#     Summarizes the uploaded PDF file.
#     Expects a PDF file to be uploaded as a POST request.
#     """
#     try:
#         # Check if a file is uploaded
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400

#         file = request.files['file']
#         original_filename = file.filename

#         # Extract text from the PDF
#         text = parse_pdf_to_text(file)

#         # Optionally save the extracted text to a temporary file
#         temp_folder = "/tmp/pdf_texts"  # Define a temporary folder for saving text files
#         temp_file_path = save_text_to_temp_file(text, temp_folder, original_filename)

#         # Load summarization model
#         model_name = "t5-small"  # Default model
#         summarizer = load_summarizer(model_name)

#         if summarizer is None:
#             return jsonify({"error": "Model not supported"}), 500

#         # Split text into chunks
#         chunks = split_text_with_langchain(text, chunk_size=4096, chunk_overlap=200)

#         # Summarize each chunk
#         summaries = []
#         for chunk in chunks:
#             try:
#                 summary = summarizer(chunk)
#                 summaries.append(chunk)
#             except Exception as e:
#                 return jsonify({"error": f"Error summarizing chunk: {e}"}), 500

#         # Combine all summaries
#         final_summary = " ".join(summaries)
#         return jsonify({"summary": final_summary, "temp_file_path": temp_file_path})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500




@app.route('/addToDynamo', methods=['POST'])
def addToDynamo():
    try:

        with open('pdf_metadata.json', 'r') as file:
            pdf_metadata = json.load(file)


        # Access the DynamoDB table
        table = dynamodb.Table(TABLE_NAME)

        for cleaned_filename, metadata in pdf_metadata.items():
            data_to_add = {
                "PaperTxtName": cleaned_filename + ".txt",  # Use the cleaned filename as the primary key
                "PaperID": metadata[0],  # First element of the list
                "PaperLink": metadata[1],  # Second element of the list
                "PaperPDFName": metadata[2]  # Third element of the list
            }
            # Add the item to the table
            table.put_item(Item=data_to_add)
            break

        # Return success response
        return jsonify({"message": "All items added successfully!"}), 200


    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500


@app.route('/getFromDynamo', methods=['POST'])
def getFromDynamo():
    try:
        # Parse the incoming JSON request to get the list of PaperIDs
        request_data = request.get_json()
        paper_ids = request_data.get("PaperIDs")

        if not paper_ids or not isinstance(paper_ids, list):
            return jsonify({"error": "Invalid input. Please provide a list of PaperIDs."}), 400

        # Access the DynamoDB table
        table = dynamodb.Table(TABLE_NAME)

        # Retrieve data for each PaperID
        results = []
        for paper_id in paper_ids:
            # Query the table for the given PaperID
            response = table.scan(
                FilterExpression=Attr("PaperID").eq(paper_id)
            )

            # Append the retrieved items to the results list
            if 'Items' in response and response['Items']:
                results.extend(response['Items'])

        # Return the results
        return jsonify({"data": results}), 200

    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
