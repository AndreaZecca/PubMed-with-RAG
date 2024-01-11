from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.retrievers import PubMedRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import dotenv
from pathlib import Paths


templates_path = Path('./templates')

def create_chain(model_id, model_kwargs, opt4=True):
    if opt4:
        with open(templates_path / 'medqa_opt4.txt') as f:
            template = f.readlines()
    else:
        with open(templates_path / 'medqa_opt5.txt') as f:
            template = f.readlines()

    prompt = PromptTemplate.from_template(template)

    retriever = PubMedRetriever()

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        device='auto',
        task="text-generation",
        pipeline_kwargs=model_kwargs,
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
    )
        


if __name__ == "main":
    dotenv.load('./env')