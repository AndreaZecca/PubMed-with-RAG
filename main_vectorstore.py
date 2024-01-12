from langchain.schema import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.retrievers import PubMedRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader

import spacy
from dotenv import load_dotenv
from pathlib import Path
from utils import get_template, load_entity_linker
from parse_dataset import parse_dataset, get_index_from_res

load_dotenv()

loader = TextLoader("../../modules/state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

vector_db = Milvus.from_documents(
    docs,
    embeddings,
    connection_args={"host": "127.0.0.1", "port": "19530"},
)

query = "What did the president say about Ketanji Brown Jackson"
docs = vector_db.similarity_search(query)