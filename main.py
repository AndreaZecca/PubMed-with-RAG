from langchain.schema import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.retrievers import PubMedRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import HuggingFaceEmbeddings


from dotenv import load_dotenv
from pathlib import Path
from utils import get_template
from parse_dataset import parse_dataset, get_index_from_res

load_dotenv()

templates_path = Path('./templates')

usr_template = None

models = [ "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.1" ]
datasets = [ "medqa_opt4", "medqa_opt5", "medncqa_opt4" ][:2]



def format_docs(docs):
    with open('log_docs.txt', 'w') as f:
        f.write(str(docs) + '\n')
    return "\n\n".join([d.page_content for d in docs])

model_id = models[2]
llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        device=0,
        task="text-generation",
        model_kwargs={"torch_dtype": "auto", "temperature": 1e-10, "do_sample": True},
        pipeline_kwargs={"max_new_tokens": 100},
)

def test_dataset(dataset):
    collection_name = 'medqa' if dataset in ['medqa_opt4', 'medqa_opt5'] else 'medncqa_opt4'

    with open(get_template(model_id, dataset)) as f:
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

    for q in questions[:10]:
        res = chain.invoke(q["question"])
        print(f"Response : {res} \n\n")
        index = get_index_from_res(res)
        if index == q["answer_idx"]:
            correct += 1
            print("The answer is correct")
        else:
            model_answer = index if index != "F" else 'No answer'
            if model_answer == "No answer":
                print(res)
    
            print(f"The answer is wrong, model answer: {model_answer}, correct {q['answer_idx']}")


for dataset in datasets:
    print(f"Testing dataset {dataset}")
    test_dataset(dataset)

