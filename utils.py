model_info = 'templates/{model_name}/{dataset_name}'

model_path_dict = {
    'HuggingFaceH4/zephyr-7b-beta': 'zephyr',
    'meta-llama/Llama-2-7b-chat-hf': 'llama',
    'mistralai/Mistral-7B-Instruct-v0.1': 'mistral'
}

dataset_path_dict = {
    'medqa_opt4': 'medqa_opt4.txt',
    'medqa_opt5': 'medqa_opt5.txt',
    'medmcqa_opt4': 'medmcqa_opt4.txt'
}


def get_template(model_name, dataset_name):
    return model_info.format(model_name=model_path_dict[model_name], dataset_name=dataset_path_dict[dataset_name])