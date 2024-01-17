import os
import json

"""
def get_index_1(response):
    import re 
    pattern = r"(the correct)?\s?answer is: (\(A\)|\(B\)|\(C\)|\(D\)|\(E\))"
    match = re.findall(pattern, response)

    if len(match) == 0:
        return "F"
    
    return match[0].replace("answer is: (", "").replace(")","")
"""


def get_index_1(response):
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


# def get_index_2(response):
#     import re 
#     pattern = r"(\(A\)|\(B\)|\(C\)|\(D\)|\(E\))"
#     match = re.findall(pattern, response)

#     if len(match) == 0:
#         return "F"
    
#     return match[0].replace("(", "").replace(")","")


# def get_index_3(response):
#     import re 
#     pattern = r"(?:The correct\s)?answer is\:? (\(A\)|\(B\)|\(C\)|\(D\)|\(E\))"
#     match = re.findall(pattern, response)

#     if len(match) == 0:
#         return "F"
    
#     return match[0].replace("(", "").replace(")","")



total_count = 0
total_f = 0
for model_name in os.listdir('.'):
    if model_name.startswith('.') or model_name.startswith('..') or model_name.endswith('.py'):
        continue
    print(f'Checking {model_name}...')
    for results_name in os.listdir(f'./{model_name}/'):
        count = 0
        if results_name.startswith('.') or results_name.startswith('..') or results_name.endswith('.py') : #or 'noRAG' in results_name
            continue
        print(f'Checking {results_name}...')
        results = None
        with open(f'./{model_name}/{results_name}', 'r') as f:
            results = json.load(f)
            for question in results['questions']:
                response = question['response']
                index = get_index_1(response)
                question['response_idx'] = index

                if index == question['answer_idx']:
                    count = count + 1   

            new_accuracy = count / results['tot_questions']
            results['accuracy'] = new_accuracy
            results['right_answers'] = count
        with open(f'./{model_name}/{results_name}', 'w') as f:
            json.dump(results, f, indent=4)         