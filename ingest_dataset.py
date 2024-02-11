from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from dotenv import load_dotenv

from milvus import default_server
from pymilvus import connections, utility, Collection


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
        utility.drop_collection("textbooks")
        print("old textbooks collection dropped")
    except:
        pass
    print("Ingesting textbooks...")
    documents = []
    for file in tqdm(glob.glob(os.path.join(file_path, "*.txt"))):
        with open(file, 'r') as f:
            data = f.read()
            documents.append(Document(page_content=data, metadata={'source': "medqa"}))

    docs = text_splitter.split_documents(documents)
    _ = Milvus.from_documents(
        docs,
        embedder,
        connection_args={"host":  "127.0.0.1", "port": "19530"}, collection_name="textbooks"
    )
    
def ingest_medwiki():
    try: 
        utility.drop_collection("medwiki")
        print("old medwiki collection dropped")
    except:
        pass
    print("Ingesting medwiki")

    from datasets import load_dataset
    dataset = load_dataset("VOD-LM/medwiki", split="train", trust_remote_code=True)
    length = len(dataset)
    max_per_iter = 5_000

    vector_db = Milvus(
        embedder,
        connection_args={"host": "127.0.0.1", "port": "19530"}, collection_name="medwiki")
    
    for i in tqdm(range(0, length, max_per_iter)):
        documents = []
        for j in range(i, min(i + max_per_iter, length)):
            document = dataset[j]
            text = document['document.title'] + document['document.text']
            documents.append(Document(page_content=text, metadata={'source': "medwiki"}))
        docs = text_splitter.split_documents(documents)
        vector_db.add_documents(docs)

def ingest_artificial():
    print('Renaming medwiki to medwiki_artificial_complete')
    utility.rename_collection("medwiki", "medwiki_artificial_complete")

    vector_db = Milvus(
        embedder,
        connection_args={"host": "127.0.0.1", "port": "19530"}, collection_name="medwiki_artificial_complete")

    print('Ingesting DEV_medmcqa_artificial')
    DEV_medmcqa_artificial = json.load(open('datasets/context/artificial/DEV_medmcqa_artificial_ctxs.json'))
    DEV_medmcqa_artificial_docs = [ctx['text'] for item in DEV_medmcqa_artificial for ctx in item['ctxs']]
    DEV_medmcqa_artificial_docs = [Document(page_content=doc, metadata={'source': "artificial"}) for doc in DEV_medmcqa_artificial_docs]
    DEV_medmcqa_artificial_docs = text_splitter.split_documents(DEV_medmcqa_artificial_docs)
    vector_db.add_documents(DEV_medmcqa_artificial_docs)

    print('Ingesting TEST_medqa_4op_artificial')
    TEST_medqa_4op_artificial = json.load(open('datasets/context/artificial/TEST_medqa_4op_artificial_ctxs.json'))
    TEST_medqa_4op_artificial_docs = [ctx['text'] for item in TEST_medqa_4op_artificial for ctx in item['ctxs']]
    TEST_medqa_4op_artificial_docs = [Document(page_content=doc, metadata={'source': "artificial"}) for doc in TEST_medqa_4op_artificial_docs]
    TEST_medqa_4op_artificial_docs = text_splitter.split_documents(TEST_medqa_4op_artificial_docs)
    vector_db.add_documents(TEST_medqa_4op_artificial_docs)

    print('Ingesting TRAIN_FID_medqa_4op_ABCD_DEF')
    TRAIN_FID_medqa_4op_ABCD_DEF = json.load(open('datasets/context/artificial/TRAIN_FID_medqa_4op_ABCD_DEF.json'))
    TRAIN_FID_medqa_4op_ABCD_DEF_docs = [ctx['text'] for item in TRAIN_FID_medqa_4op_ABCD_DEF for ctx in item['ctxs']]
    TRAIN_FID_medqa_4op_ABCD_DEF_docs = [Document(page_content=doc, metadata={'source': "artificial"}) for doc in TRAIN_FID_medqa_4op_ABCD_DEF_docs]
    TRAIN_FID_medqa_4op_ABCD_DEF_docs = text_splitter.split_documents(TRAIN_FID_medqa_4op_ABCD_DEF_docs)
    vector_db.add_documents(TRAIN_FID_medqa_4op_ABCD_DEF_docs)

    print('Ingesting TRAIN_FID_medmcqa_100percent')
    TRAIN_FID_medmcqa_100percent = json.load(open('datasets/context/artificial/TRAIN_FID_medmcqa_100percent.json'))
    TRAIN_FID_medmcqa_100percent_docs = [ctx['text'] for item in TRAIN_FID_medmcqa_100percent for ctx in item['ctxs']]
    total_length = len(TRAIN_FID_medmcqa_100percent_docs)
    max_per_iter = 10_000
    for i in tqdm(range(0, total_length, max_per_iter)):
        documents = []
        for j in range(i, min(i + max_per_iter, total_length)):
            documents.append(Document(page_content=TRAIN_FID_medmcqa_100percent_docs[j], metadata={'source': "artificial"}))
        docs = text_splitter.split_documents(documents)
        vector_db.add_documents(docs)

    

def ingest_all():
    raise NotImplementedError("Not implemented yet")

@click.command()
@click.option("--dataset", help="Dataset", required=True, type=click.Choice(["medqa", 'medwiki', "artificial", "all"]), default="all")
def main(dataset):    
    if dataset == "medqa":
        ingest_medqa("datasets/context/medqa/")
    elif dataset == 'medwiki':
        ingest_medwiki()
    elif dataset == "artificial":
        ingest_artificial()
    elif dataset == "all":
        ingest_all()
    else:
        raise ValueError("Invalid dataset")

if __name__ == "__main__":
    main()
