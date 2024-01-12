from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from dotenv import load_dotenv

from milvus import default_server
from pymilvus import connections


#default_server.start()
#connections.connect("default", host="0.0.0.0")

load_dotenv()

embedder = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def ingest_txt(file_path: str):
    loader = TextLoader(file_path)
    documents = loader.load()

    print("Splitting documents...")
    docs = text_splitter.split_documents(documents)

    print("Ingesting documents...")
    vector_db = Milvus.from_documents(
        docs,
        embedder,
        connection_args={"host":  "127.0.0.1", "port": "19530"}, collection_name="medmcqa"
    )

def ingest_json(file_path: str):
    import json
    with open(file_path, "r") as f:
        data = json.load(f)

    print("Splitting documents...")

    contexts = []
    for item in data:
        contexts.extend(item)

    documents = []
    for context in contexts:
        documents.append(Document(page_content=context, metadata={'source': file_path}))

    print("Splitting documents...")
    docs = text_splitter.split_documents(documents)

    print("Ingesting documents...")
    vector_db = Milvus.from_documents(
        docs,
        embedder,
        connection_args={"host":  "127.0.0.1", "port": "19530"}, collection_name="medqa"
    )


def ingest_all():
    from pymilvus import connections, utility

    try:
        conn = connections.connect(
            alias="default",
            user='username',
            password='password',
            host="127.0.0.1",
            port='19530'
        )

        input("Press enter to drop collection")

        utility.drop_collection("medmcqa")
        utility.drop_collection("medqa")
        print("Dropped collection")
    except:
        print("Collection doesn't exist")

    import glob
    import os

    textbooks_folder_path: str = "datasets/context/medqa/"
    files = glob.glob(os.path.join(textbooks_folder_path, "*.txt"))

    for file in files:
        print(f"Reading file {file}")
        ingest_txt(file)

    print("Reading json...")
    ingest_json("datasets/context/medmcqa/context.json")

input("Press enter to ingest all")
ingest_all()