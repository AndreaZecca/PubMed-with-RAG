from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import PubMedRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

import langchain
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts.prompt import PromptTemplate

from pathlib import Path

load_dotenv()

langchain.verbose = True

templates_path = Path('./templates')

# templates = {
#     "zephyr": {
#         "sys_start": '<|system|>\n',
#         "sys_end": '</s>\n',
#         "usr_start": '<|user|>\n',
#         "usr_end": '</s>\n',
#         "assistant_start": '<|assistant|>\n'
#     }
# }


# def create_prompt(model_name):
#     full_template = """
#     {system}

#     {usr_msg}
#     """
#     model_template = templates[model_name]
#     full_prompt = PromptTemplate.from_template(full_template)
#     introduction_template = f"{model_template['sys_start']}You are a medical expert. Just answer the question as concise as possible based on the user needs. Always return the response in the expected format required by the user. {model_template['sys_end']}"
#     introduction_prompt = PromptTemplate.from_template(introduction_template)

#     with open(templates_path / 'medqa_opt4.txt') as f:
#         usr_template = f.read()

#     usr_template = usr_template.format(
#         usr_start=model_template['usr_start'],
#         usr_end=model_template['usr_end'],
#         assistant_start=model_template['assistant_start']
#     )
#     usr_prompt = PromptTemplate.from_template(usr_template)


#     input_prompts = [
#     ("system", introduction_prompt),
#     ("usr_msg", usr_prompt)]

#     pipeline_prompt = PipelinePromptTemplate(
#     final_prompt=full_prompt, pipeline_prompts=input_prompts)

#     #print(pipeline_prompt.input_variables)
#     return pipeline_prompt



    

# def create_chain(model_id, model_kwargs, opt4=True):

#     prompt = create_prompt("zephyr")
#    # print('Prompt')

#     retriever = PubMedRetriever()

#     llm = HuggingFacePipeline.from_model_id(
#         model_id=model_id,
#         device=0,
#         task="text-generation",
#         model_kwargs=model_kwargs,
#         pipeline_kwargs={"max_new_tokens": 100},
#     )

#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)


#     rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
#     )
#     response = rag_chain.invoke("""A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?
#     (A) Disclose the error to the patient and put it in the operative report
#     (B) Tell the attending that he cannot fail to disclose this mistake
#     (C) Report the physician to the ethics committee
#     (D) Refuse to dictate the operative report
#     """
#     )
#     print(response)
        



#create_chain("HuggingFaceH4/zephyr-7b-beta", {"torch_dtype": "auto"})

from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

usr_template = None

with open(templates_path / 'medqa_opt4.txt') as f:
        usr_template = f.read()

template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(usr_template)

llm = HuggingFacePipeline.from_model_id(
        model_id="meta-llama/Llama-2-7b-chat-hf",#"mistralai/Mistral-7B-Instruct-v0.1",#"HuggingFaceH4/zephyr-7b-beta",
        device=0,
        task="text-generation",
        model_kwargs={"torch_dtype": "auto"},
        pipeline_kwargs={"max_new_tokens": 1000},
    )

retriever = PubMedRetriever()

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
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