from langchain.schema import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.retrievers import PubMedRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough

import spacy
from dotenv import load_dotenv
from pathlib import Path
from utils import get_template, load_entity_linker
from parse_dataset import parse_dataset, get_index_from_res

load_dotenv()

templates_path = Path('./templates')

usr_template = None

models = [ "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.1" ]
datasets = [ "medqa4", "medqa5", "altro_dataset" ]

model_id = models[2]
dataset = datasets[0]

with open(get_template(model_id, dataset)) as f:
        usr_template = f.read()

prompt = ChatPromptTemplate.from_template(usr_template)

llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        device=0,
        task="text-generation",
        model_kwargs={"torch_dtype": "auto", "temperature": 1e-10, "do_sample": True},
        pipeline_kwargs={"max_new_tokens": 100},
    )

retriever = PubMedRetriever(search_type="similarity_score_threshold", top_k_results=5, score_thresholds=0.1, search_kwargs={"score_threshold": 0.01})

def format_docs(docs):
    with open('log_docs.txt', 'w') as f:
        f.write(str(docs) + '\n')
    return "\n\n".join([d.page_content for d in docs])

nlp = load_entity_linker()

def print_query(query):
    print("QUERY:", query)
    return query

def format_query(query):
    # take only from (A)
    query = query.split("(A)")[1]

    doc = nlp(query)
    print("DOC:",doc)
    with open('log_query.txt', "w") as f:
        full_text = ""
        full_labels = ""
        for ent in doc.ents:
            full_labels += ent.label_ + " "
            full_text += ent.text + " "
        full_text += "\n"
        full_labels += "\n"
        f.write(full_text)
        f.write(full_labels)
    # Qui andrebbe la formattazione / ricerca delle keywords
    return full_text


chain = (
    {"context": format_query | retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = parse_dataset("medqa_4opt.jsonl")

for q in questions[:1]:
    res = chain.invoke(q["question"])
    print(res)
    index = get_index_from_res(res)
    if index == q["answer_idx"]:
        print("Correct!")
    else:
        print("Wrong!")