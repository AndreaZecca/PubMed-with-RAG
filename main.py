from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings

from pathlib import Path
from dotenv import load_dotenv
import json
from tqdm import tqdm
import click

import warnings
warnings.filterwarnings("ignore", message="")


from utils import get_template, get_results_path
from parse_dataset import parse_dataset, get_index_from_res

load_dotenv()

templates_path = Path('./templates')

usr_template = None


global_context = []

def format_docs(docs):
    global global_context
    global_context = [d.page_content for d in docs]
    return "\n\n".join([d.page_content for d in docs])

def test_dataset(dataset, llm, model):
    collection_name = 'medqa' if dataset in ['medqa_opt4', 'medqa_opt5'] else 'medmcqa'

    with open(get_template(model, dataset)) as f:
        usr_template = f.read()

    prompt = ChatPromptTemplate.from_template(usr_template)

    embedder = HuggingFaceEmbeddings(model_name="neuml/pubmedbert-base-embeddings")

    vector_db = Milvus(
        embedder,
        connection_args={"host": "127.0.0.1", "port": "19530"},
        collection_name=collection_name,
    )
    
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vector_db.as_retriever()

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = parse_dataset(f"{dataset}.jsonl")

    correct = 0
    toal_questions = 5#len(questions)

    all_questions = []
    
    for q in tqdm(questions[:5]):      
        question = q["question"]
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
        "tot_questions": toal_questions,
        "right_answers": correct,
        "accuracy": correct / toal_questions
    }
    with open(get_results_path(model, dataset), "w") as f:
        json.dump(output_json, f)    


@click.command()
@click.option("--model", help="Model to use", required=True)
@click.option("--datasets", help="Dataset to use", required=True)
def main(model, datasets):
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
        test_dataset(dataset, llm, model)
        print()

if __name__ == "__main__":
    main()
