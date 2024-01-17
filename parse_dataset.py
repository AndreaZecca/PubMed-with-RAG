import json

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


def parse_dataset(dataset_name: str) -> list[dict]:
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
    pattern = r"\(A\)|\(B\)|\(C\)|\(D\)|\(E\)"
    match = re.findall(pattern, response)

    if len(match) == 0:
        return "F"

    
    if len(match) > 1:
        pattern = r"(?:The correct\s)?answer is:? (\(A\)|\(B\)|\(C\)|\(D\)|\(E\))"
        match = re.findall(pattern, response)
    
        if len(match) == 0:
            return "F"

        return match[0].replace("(", "").replace(")","")

    return match[0].replace("(", "").replace(")","")