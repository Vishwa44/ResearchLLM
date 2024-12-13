import os
from ollama import generate, embed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from time import time
from pathlib import Path


pinecone_chunks = []

file_names = []


def chunk_text_from_file(file_path, chunk_size=500, chunk_overlap=50):
    # Convert to Path object for better path handling
    file_path = Path(file_path)
    
    # Verify file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    loader = TextLoader(str(file_path), encoding='utf-8')  # Explicitly specify encoding
    document = loader.load()
    print(f"Successfully loaded: {file_path}")
    
    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(document)
    return chunks


def get_doc_chunks_embeddings(document_path, chunks):
    chunks_with_context = []
    print(f"Num of chunks: {len(chunks)}")

    try:
        with open(document_path, 'r', encoding='utf-8') as file:
            document = file.read()
    except Exception as e:
        print(f"Error reading file {document_path}: {str(e)}")
        return []
    


    context_time_1 = time()
    for chunk in chunks:
    
        prompt = f""""
        <document>
        {document}
        </document> 
        Here is the chunk we want to situate within the whole document 
        <chunk> 
        {chunk.page_content}
        </chunk> 
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 
        """

        # generate context for chunk
        chunk_context = generate(model="llama3.2", prompt=prompt).response

        # append the context to the chunk and generate embeddings
        chunk_with_context = chunk_context + chunk.page_content
        chunks_with_context.append(chunk_with_context)

    context_time_2 = time()
    print(f"Time taken to generate context strings: {context_time_2 - context_time_1} seconds")

    t1 = time()
    embeddings = embed(model="llama3.2", input=chunks_with_context)
    t2 = time()
    print(f"Time taken to generate embeddings: {t2 - t1} seconds")
    return embeddings.embeddings


# Store embeddings and paper IDs in FAISS index
def gen_embedding_dataset(folder_path):
    
    all_embeddings = []
    paper_ids = []
    pinecone_chunks = []
    folder_path = Path(folder_path)
    try:
    
        for file_number, file_name in enumerate(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)
            file_names.append(file_name)
            
            if os.path.isfile(file_path) and file_name.endswith('.txt'):
                paper_id = "paper_" + str(file_number+2200)
                print(file_path)
                # paper_id = os.path.basename(file_path)
                chunks = chunk_text_from_file(file_path)
                embeddings = get_doc_chunks_embeddings(file_path, chunks)
                for i, _ in enumerate(chunks):
                    temp_dict = {}
                    temp_dict["id"] = paper_id + "#chunk_" + str(i)
                    temp_dict["values"] = embeddings[i]                
                    temp_dict["metadata"] = {"paper_id": paper_id, "chunk_id": "chunk_" + str(i), "chunk": chunks[i].page_content}
                    pinecone_chunks.append(temp_dict)

                    # temp_dict["paper_id"] = paper_id
                    # temp_dict["chunk_id"] = "chunk_" + str(i)
                    # temp_dict["chunk"] = chunks[i].page_content
                all_embeddings.extend(embeddings)
                paper_ids.extend([paper_id] * len(embeddings))


    except Exception as e:
        print(f"Error occurred: {e}")
        return None
    
    finally:
        return pinecone_chunks

if __name__ == "_main_":
    try:

        # Method 1: Using raw string
        # path = Path(r"C:\Users\satya\cc_proj\parsed_dataset")

        # Method 2: Using forward slashes (automatically converted)
        path = Path("C:/Users/satya/cc_proj/parsed_dataset/")

        
        pinecone_chunks = gen_embedding_dataset(path)
        print(file_names)



    except Exception as e:
        print(f"Error occurred: {e}")