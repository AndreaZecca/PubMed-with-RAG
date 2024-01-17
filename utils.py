model_info = 'templates/{model_name}/{dataset_name}'

results_info = 'results/{model_name}/{dataset_name}'

model_path_dict = {
    'HuggingFaceH4/zephyr-7b-beta': 'zephyr',
    'meta-llama/Llama-2-7b-chat-hf': 'llama',
    'mistralai/Mistral-7B-Instruct-v0.1': 'mistral'
}

dataset_path_dict = {
    'medqa_opt4': 'medqa_opt4.txt',
    'medqa_opt5': 'medqa_opt5.txt',
    'medmcqa_opt4': 'medmcqa_opt4.txt',
    'mmlu_opt4': 'mmlu_opt4.txt'
}

results_path_dict = {
    'medqa_opt4': 'medqa_opt4.json',
    'medqa_opt5': 'medqa_opt5.json',
    'medmcqa_opt4': 'medmcqa_opt4.json',
    'mmlu_opt4': 'mmlu_opt4.json'
}

def get_template(model_name, dataset_name, rag):
    model_name_value = model_path_dict[model_name]
    dataset_name_value = dataset_path_dict[dataset_name]
    if not rag:
        dataset_name_value = dataset_name_value.replace('.txt', '_noRAG.txt')
    return model_info.format(model_name=model_name_value, dataset_name=dataset_name_value)

def get_results_path(model_name, dataset_name, optional_path=''):
    return results_info.format(model_name=model_path_dict[model_name] + optional_path, dataset_name=results_path_dict[dataset_name])