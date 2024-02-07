from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from dotenv import load_dotenv

from milvus import default_server
from pymilvus import connections, utility


import click
import glob
import os
from tqdm import tqdm
import json

try:
    connections.connect("default", host="0.0.0.0")
except:
    default_server.start()
    connections.connect("default", host="0.0.0.0")

load_dotenv()

embedder = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1_000, chunk_overlap=200)

def ingest_medqa(file_path: str):
    try:
        utility.drop_collection("medqa")
        print("old medqa collection dropped")
    except:
        pass
    print("Ingesting medqa...")
    documents = []
    for file in tqdm(glob.glob(os.path.join(file_path, "*.txt"))):
        with open(file, 'r') as f:
            data = f.read()
            documents.append(Document(page_content=data, metadata={'source': "medqa"}))

    docs = text_splitter.split_documents(documents)
    _ = Milvus.from_documents(
        docs,
        embedder,
        connection_args={"host":  "127.0.0.1", "port": "19530"}, collection_name="medqa"
    )
    
def ingest_medmcqa_mmlu():
    try: 
        utility.drop_collection("medmcqa_mmlu")
        print("old medmcqa_mmlu collection dropped")
    except:
        pass
    print("Ingesting medmcqa and mmlu...")

    from datasets import load_dataset
    dataset = load_dataset("VOD-LM/medwiki", split="train", trust_remote_code=True)
    length = len(dataset)
    max_per_iter = 5_000

    vector_db = Milvus(
        embedder,
        connection_args={"host": "127.0.0.1", "port": "19530"}, collection_name="medmcqa_mmlu" )
    
    for i in tqdm(range(0, length, max_per_iter)):
        documents = []
        for j in range(i, min(i + max_per_iter, length)):
            document = dataset[j]
            text = document['document.title'] + document['document.text']
            documents.append(Document(page_content=text, metadata={'source': "medmcqa"}))
        docs = text_splitter.split_documents(documents)
        vector_db.add_documents(docs)


def ingest_all():
    ingest_medqa("datasets/context/medqa/")
    ingest_medmcqa_mmlu()

@click.command()
@click.option("--dataset", help="Dataset", required=True, type=click.Choice(["medqa", "medmcqa", "mmlu", "all"]), default="all")
def main(dataset):    
    if dataset == "all":
        ingest_all()
    elif dataset == "medqa":
        ingest_medqa("datasets/context/medqa/")
    elif dataset in ["medmcqa", 'mmlu']:
        ingest_medmcqa_mmlu()
    else:
        raise ValueError("Invalid dataset")

if __name__ == "__main__":
    main()
