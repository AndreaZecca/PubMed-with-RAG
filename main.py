import langchain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.retrievers import PubMedRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv

from pathlib import Path

from utils import get_template

load_dotenv()

langchain.verbose = True

templates_path = Path('./templates')

usr_template = None

models = [ "HuggingFaceH4/zephyr-7b-beta", "meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.1" ]
datasets = [ "medqa4", "medqa5", "altro_dataset" ]

model_id = models[0]
dataset = datasets[0]

with open(get_template(model_id, dataset)) as f:
        usr_template = f.read()

prompt = ChatPromptTemplate.from_template(usr_template)

llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        device=0,
        task="text-generation",
        model_kwargs={"torch_dtype": "auto", "temperature": 0.00001, "do_sample": True},
        pipeline_kwargs={"max_new_tokens": 100},
    )

retriever = PubMedRetriever(search_type="similarity", top_k_results=5, score_thresholds=0.5)

def format_docs(docs):
    with open('log.txt', 'w') as f:
        f.write(str(docs) + '\n')
    return "\n\n".join([d.page_content for d in docs])

query_only_answers = True

def format_query(query):
    # Qui andrebbe la formattazione / ricerca delle keywords
    return query


chain = (
    {"context": format_query | retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke("""A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?  
    (A) Disclose the error to the patient and put it in the operative report
    (B) Tell the attending that he cannot fail to disclose this mistake
    (C) Report the physician to the ethics committee
    (D) Refuse to dictate the operative report
    
    """
    )

print(res)