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

from utils import get_template, get_results_path, parse_dataset, get_index_from_res

load_dotenv()

templates_path = Path('./templates')

usr_template = None
RETRIEVE_WITH_RERANK = 10
KEEP_TOP = 5


global_context = []
global_query = None
global_debug = False
global_rerank = False
global_rag = True


embedder = None 
reranker_model = None


def format_docs(docs):
    return "\n".join([d.page_content for d in docs])

def rerank_docs(docs):
    global global_query, global_context
    # rerank docs
    pairs = [list((global_query, d.page_content)) for d in docs]
    scores = reranker_model.compute_score(pairs)
    permutation = np.argsort(scores)[::-1]
    docs = [docs[i] for i in permutation]
    global_context = [d.page_content for d in docs]
    docs = docs[:KEEP_TOP]
    return docs

def process_docs(docs):
    global global_rerank, global_context
    if global_rerank:
        docs = rerank_docs(docs)
    else:
        global_context = [d.page_content for d in docs]
    return format_docs(docs)

# def debug_prompt(prompt):
#     global global_debug
#     if global_debug:
#         with open("debug_prompt.txt", "w") as f:
#             f.write(prompt.messages[0].content)
#     return prompt

def test_dataset(dataset, llm, model, rag, collection=None):
    global global_debug, global_rerank, global_query, embedder, reranker_model
    
    collection_name = 'medqa' if dataset in ['medqa_opt4', 'medqa_opt5'] else 'medmcqa_mmlu'

    if collection:
        collection_name = collection

    with open(get_template(model, dataset, rag)) as f:
        usr_template = f.read()
    
    prompt = ChatPromptTemplate.from_template(usr_template)
    
    if rag:
        vector_db = Milvus(
            embedder,
            connection_args={"host": "127.0.0.1", "port": "19530"},
            collection_name=collection_name,
        )
        retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVE_WITH_RERANK if global_rerank else KEEP_TOP})

        base_chain = {"context": retriever | process_docs, "question": RunnablePassthrough()}
    else:
        base_chain = {"question": RunnablePassthrough()}

    chain = (
        base_chain
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = parse_dataset(f"{dataset}.jsonl")

    if global_debug:
        questions = questions[:3]

    correct = 0
    total_questions = len(questions)
    
    all_questions = []
    
    for q in tqdm(questions):      
        question = q["question"]
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
        
    if not rag:
        result_path = result_path.replace('.json', '_noRAG.json')
    
    if global_rerank:
        result_path = result_path.replace('.json', '_rerank.json')
    
    if collection:
        result_path = result_path.replace('.json', f'_{collection}.json')
    
    if global_debug:
        result_path = result_path.replace('.json', '_debug.json')

    with open(result_path, "w") as f:
        json.dump(output_json, f, indent=4)   
    
    return correct / total_questions 


@click.command()
@click.option("--model", help="Model to use", required=True)
@click.option("--datasets", help="Dataset to use", required=True)
@click.option("--rerank", help="Whether to use reranking", required=False, default=False)
@click.option("--debug", help="Debug mode", required=False, default=False)
@click.option("--rag", help="Whether to use RAG", required=False, default=True)
@click.option("--collection", help="Collection to use", required=False, default=None)
def main(model, datasets, rerank, debug, rag, collection):
    global global_debug, global_rerank, global_rag, embedder, reranker_model
    global_debug = debug
    global_rerank = rerank
    datasets = datasets.split(',')
    model_kwargs={"torch_dtype": "auto", "do_sample": False, 'pad_token_id': 0}

    force_medwiki = True

    if rag:
        embedder = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")
        if rerank:
            reranker_model = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
    else:
        global_rag = False
        global_rerank = False

    llm = HuggingFacePipeline.from_model_id(
            model_id=model,
            device=0,
            task="text-generation",
            model_kwargs=model_kwargs,
            pipeline_kwargs={"max_new_tokens": 50},
    )
    for dataset in datasets:
        print(f'Testing model {model} with dataset {dataset} ', end='')
        if rag:
            print('using RAG ', end='')
        if rerank:
            print('with reranking ', end='')
        if debug:
            print(' in debug mode ', end='')
        print('\n')

        accuracy = test_dataset(dataset, llm, model, rag, collection)
        print(f'Accuracy: {accuracy:.2f}%')
        print()

if __name__ == "__main__":
    main()
