from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings

from FlagEmbedding import FlagReranker

from pathlib import Path
from dotenv import load_dotenv
import json
from tqdm import tqdm
import click
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="")

import transformers
transformers.logging.set_verbosity_error()

from utils import get_template, get_results_path
from parse_dataset import parse_dataset, get_index_from_res

load_dotenv()

templates_path = Path('./templates')

usr_template = None
RETRIEVE_WITH_RERANK = 10
KEEP_TOP = 5


global_context = []
global_query = None

embedder = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")
reranker_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True) # use_fp16 speeds up computation with a slight performance degradation

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def rerank_docs(docs):
    global global_context
    global global_query

    # rerank docs
    pairs = [list((global_query, d.page_content)) for d in docs]
    scores = reranker_model.compute_score(pairs)
    permutation = np.argsort(scores)[::-1]
    docs = [docs[i] for i in permutation]
    global_context = [d.page_content for d in docs]
    docs = docs[:KEEP_TOP]
    return docs

def test_dataset(dataset, llm, model, rerank, debug):
    collection_name = 'medqa' if dataset in ['medqa_opt4', 'medqa_opt5'] else 'medmcqa'

    with open(get_template(model, dataset)) as f:
        usr_template = f.read()

    prompt = ChatPromptTemplate.from_template(usr_template)

    vector_db = Milvus(
        embedder,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        collection_name=collection_name,
    )
    
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVE_WITH_RERANK if rerank else KEEP_TOP})

    if rerank:
        base_chain = {"context": retriever | rerank_docs | format_docs, "question": RunnablePassthrough()}
    else:
        base_chain = {"context": retriever | format_docs, "question": RunnablePassthrough()}

    chain = (
        base_chain
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = parse_dataset(f"{dataset}.jsonl")

    if debug:
        questions = questions[:3]

    correct = 0
    total_questions = len(questions)
    
    all_questions = []
    
    for q in tqdm(questions):      
        question = q["question"]
        global global_query
        global_query = question
        response = chain.invoke(question)
        response_idx = get_index_from_res(response)
        answer_idx = q["answer_idx"]
        context = global_context
        if response_idx == answer_idx:
            correct += 1
        all_questions.append({
            "question": question,
            "response": response,
            "response_idx": response_idx,
            "answer_idx": answer_idx,
            "context": context
        })
    output_json = {
        "questions": all_questions,
        "tot_questions": total_questions,
        "right_answers": correct,
        "accuracy": correct / total_questions
    }
    result_path = get_results_path(model, dataset)
    if debug:
        result_path = result_path.replace('.json', '_debug.json')
    if rerank:
        result_path = result_path.replace('.json', '_rerank.json')
    with open(result_path, "w") as f:
        json.dump(output_json, f, indent=4)   
    return correct / total_questions 


@click.command()
@click.option("--model", help="Model to use", required=True)
@click.option("--datasets", help="Dataset to use", required=True)
@click.option("--rerank", help="Whether to use reranking", required=False, default=False)
@click.option("--debug", help="Debug mode", required=False, default=False)
def main(model, datasets, rerank, debug):
    datasets = datasets.split(',')
    model_kwargs={"torch_dtype": "auto", "temperature": 1e-16, "do_sample": True}
    if 'mistral' in model:
        model_kwargs['pad_token_id'] = 0   
    llm = HuggingFacePipeline.from_model_id(
            model_id=model,
            device=0,
            task="text-generation",
            model_kwargs=model_kwargs,
            pipeline_kwargs={"max_new_tokens": 100},
    )
    for dataset in datasets:
        print(f'Testing model {model} with dataset {dataset}')
        accuracy = test_dataset(dataset, llm, model, rerank, debug)
        print(f'Accuracy: {accuracy:.2f}%')
        print()

if __name__ == "__main__":
    main()
