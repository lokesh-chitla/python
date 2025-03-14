import os
import json
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
from io import StringIO
from kaggle.api.kaggle_api_extended import KaggleApi
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from transformers import pipeline

KAGGLE_USERNAME = "chitlalokeshkumar"
KAGGLE_KEY = "e8017c7cd0205e2fc459cce3b7d1bbb2"

def authenticate_kaggle():
    """Authenticates Kaggle API."""
    kaggle_json = {"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}
    
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        json.dump(kaggle_json, f)
    
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    
    api = KaggleApi()
    api.authenticate()
    
    return api

try:
    api = authenticate_kaggle()
    st.success("✅ Kaggle API authenticated successfully!")
except Exception as e:
    st.error(f"❌ Kaggle API authentication failed: {e}")

def download_and_extract_data(api):
    dataset = "securities-exchange-commission/financial-statement-extracts"
    download_path = "./financial-statement-extracts.zip"
    extract_path = "./financial-statement-extracts"
    api.dataset_download_files(dataset, path=".", unzip=False)
    try:
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    except zipfile.BadZipFile:
        raise ValueError("Downloaded file is not a valid ZIP archive.")
    return extract_path

def load_json_data(files, path):
    data_frames = []
    for file in files:
        try:
            with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                if 'num.txt' in json_data:
                    tsv_data = json_data['num.txt'].strip().replace('"', '"')
                    df = pd.read_csv(StringIO(tsv_data), sep='\t', engine='python', on_bad_lines='skip')
                    data_frames.append(df)
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error loading {file}: {e}")
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def preprocess_data(financial_data):
    if 'ddate' in financial_data.columns:
        financial_data['ddate'] = pd.to_datetime(financial_data['ddate'], format='%Y%m%d', errors='coerce')
        financial_data = financial_data[financial_data['ddate'].dt.year >= (pd.Timestamp.now().year - 2)]
    else:
        raise ValueError("No valid date column found in dataset.")
    return financial_data

def create_text_chunks(df):
    text_chunks = []
    for _, row in df.iterrows():
        chunk = f"Tag: {row.get('tag', 'Unknown')} | Date: {row.get('ddate', 'N/A')}\n"
        chunk += f"Value: {row.get('value', 'N/A')} | Unit: {row.get('unit', 'N/A')}\n"
        text_chunks.append(chunk)
    return text_chunks

def embed_text_chunks(chunks, model):
    return model.encode(chunks, convert_to_numpy=True)

def is_financial_query(query):
    candidate_labels = ["financial", "non-financial"]
    result = classifier(query, candidate_labels)
    return result['labels'][0] == 'financial'

def multi_stage_retrieval(query, k=5):
    if not query or len(query) > 500 or not re.match(r'^[\w\s\d.,!?$%&\'"()\-]+$', query):
        return ["Invalid query."], [0.0]
    if not is_financial_query(query):
        return ["Non-financial query detected."], [0.0]
    
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_k_bm25_indices = np.argsort(bm25_scores)[-k:][::-1]
    
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)
    faiss_distances, faiss_indices = index.search(query_embedding, k)
    
    results = [text_chunks[i] for i in top_k_bm25_indices]
    confidence_scores = [bm25_scores[i] for i in top_k_bm25_indices]
    
    return results, confidence_scores

def guardrail_multi_stage_retrieval(query, k=5):
    results, confidence_scores = multi_stage_retrieval(query, k)
    if "Non-financial query detected." in results or "Invalid query." in results:
        return results, confidence_scores
    
    filtered_results = [res for res in results if "Date" in res]
    filtered_confidence_scores = [confidence_scores[i] for i, res in enumerate(results) if res in filtered_results]
    
    if not filtered_results:
        return ["No relevant financial data found."], [0.0]
    
    return filtered_results, filtered_confidence_scores

api = initialize_kaggle_api()
extract_path = download_and_extract_data(api)
extracted_files = os.listdir(extract_path)
json_files = [f for f in extracted_files if f.endswith('.json')]
financial_data = load_json_data(json_files, extract_path)
financial_data = preprocess_data(financial_data)
text_chunks = create_text_chunks(financial_data)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_text_chunks(text_chunks, embedding_model)
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings))
tokenized_chunks = [chunk.split() for chunk in text_chunks]
bm25 = BM25Okapi(tokenized_chunks)
classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

st.title("Financial Query Retrieval System")
tab1, tab2 = st.tabs(["Basic Retrieval", "Guardrails Retrieval"])

with tab1:
    query = st.text_input("Enter your financial query", key="basic")
    if st.button("Retrieve", key="btn_basic"):
        results, confidence_scores = multi_stage_retrieval(query)
        for res, score in zip(results, confidence_scores):
            st.write(f"Result: {res}\nConfidence: {score:.2f}")

with tab2:
    query = st.text_input("Enter your financial query", key="guardrails")
    if st.button("Retrieve", key="btn_guardrails"):
        results, confidence_scores = guardrail_multi_stage_retrieval(query)
        for res, score in zip(results, confidence_scores):
            st.write(f"Result: {res}\nConfidence: {score:.2f}")
