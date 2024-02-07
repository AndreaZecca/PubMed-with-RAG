import json

model_info = 'templates/{model_name}/{dataset_name}'

results_info = 'results/{model_name}/{dataset_name}'

model_path_dict = {
    'HuggingFaceH4/zephyr-7b-beta': 'zephyr',
    'meta-llama/Llama-2-7b-chat-hf': 'llama',
    'mistralai/Mistral-7B-Instruct-v0.1': 'mistral'
}

dataset_path_dict = {
    'medmcqa_opt4': 'medmcqa_opt4.txt',
    'medqa_opt4': 'medqa_opt4.txt',
    'medqa_opt5': 'medqa_opt5.txt',
    'mmlu_opt4': 'mmlu_opt4.txt'
}

results_path_dict = {
    'medqa_opt4': 'medqa_opt4.json',
    'medqa_opt5': 'medqa_opt5.json',
    'medmcqa_opt4': 'medmcqa_opt4.json',
    'mmlu_opt4': 'mmlu_opt4.json'
}

def create_result_folder(model_name):
    from pathlib import Path
    model_name_value = model_path_dict[model_name]
    Path(f'results/{model_name_value}/').mkdir(parents=True, exist_ok=True)

def get_template(model_name, dataset_name, rag):
    model_name_value = model_path_dict[model_name]
    dataset_name_value = dataset_path_dict[dataset_name]
    if not rag:
        dataset_name_value = dataset_name_value.replace('.txt', '_noRAG.txt')
    return model_info.format(model_name=model_name_value, dataset_name=dataset_name_value)

def get_results_path(model_name, dataset_name):
    create_result_folder(model_name)
    model_name_value = model_path_dict[model_name]
    dataset_name_value=results_path_dict[dataset_name]
    return results_info.format(model_name=model_name_value, dataset_name=dataset_name_value)



def _parse_data(data):
    for item in data:
        question_str = item['question']

        question_str += "\n(A) " + item['options']['A']
        question_str += "\n(B) " + item['options']['B']
        question_str += "\n(C) " + item['options']['C']
        question_str += "\n(D) " + item['options']['D']

        if 'E' in item['options']:
            question_str += "\n(E) " + item['options']['E']

        question_str += "\n"

        yield {
            "question": question_str,
            "answer_idx": item['answer_idx'],
            "answer": item['answer'],
        }


def parse_dataset(dataset_name: str):
    """Parse dataset from json file.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        list[dict]: List of dictionaries containing the data.
    """
    
    with open(f'datasets/{dataset_name}', 'r') as f:
        data = f.readlines()
        json_data = [json.loads(item) for item in data]

    return [item for item in _parse_data(json_data)]


def get_index_from_res(response):
    import re 
    # pattern = r"\(A\)|\(B\)|\(C\)|\(D\)|\(E\)"
    # match = re.findall(pattern, response)

    # if len(match) == 0:
    #     return "F"

    
    # if len(match) > 1:
    #     pattern = r"(?:The correct\s)?answer is:? (\(A\)|\(B\)|\(C\)|\(D\)|\(E\))"
    #     match = re.findall(pattern, response)
    
    #     if len(match) == 0:
    #         return "F"

    #     return match[0].replace("(", "").replace(")","")

    # return match[0].replace("(", "").replace(")","")
    
    start = response.find('(')
    end = response.find(')')
    idx = response[start+1:end]
    if idx not in "ABCDE":
        idx = "F"
    return idx