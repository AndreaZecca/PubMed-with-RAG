import spacy
import scispacy
from scispacy.linking import EntityLinker

model_info = 'templates/{model_name}/{dataset_name}'

model_path_dict = {
    'HuggingFaceH4/zephyr-7b-beta': 'zephyr',
    'meta-llama/Llama-2-7b-chat-hf': 'llama',
    'mistralai/Mistral-7B-Instruct-v0.1': 'mistral'
}

dataset_path_dict = {
    'medqa4': 'medqa_opt4.txt',
    'medqa5': 'medqa_opt5.txt',
    'medmcqa': 'medmcqa_4opt.txt'
}


def get_template(model_name, dataset_name):
    return model_info.format(model_name=model_path_dict[model_name], dataset_name=dataset_path_dict[dataset_name])


def load_entity_linker(threshold=0.90):
    nlp = spacy.load("en_core_sci_lg")
    nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    return nlp

