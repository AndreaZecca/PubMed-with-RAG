import pandas as pd

DATA_PATH = './validation-00000-of-00001.parquet'

df = pd.read_parquet(DATA_PATH)

# drop null values
df = df.dropna()

# get rows where choice_type is 'single'
df = df[df['choice_type'] == 'single']

# get questions
questions = df['question'].values

# get options
options_a = df['opa'].values
options_b = df['opb'].values
options_c = df['opc'].values
options_d = df['opd'].values

# get correct answer
answers = df['cop'].values

answer_to_idx = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}

answer_to_text = {
    "A": options_a,
    "B": options_b,
    "C": options_c,
    "D": options_d
}


# map answers to 'A', 'B', 'C', 'D'
answers_idxs = [answer_to_idx[answer] for answer in answers]

formatted_question = []

for i, (question, opa, opb, opc, opd, answer) in enumerate(zip(questions, options_a, options_b, options_c, options_d, answers_idxs)):
    formatted_question.append({
        "question": question,
        "options": {
            "A": opa,
            "B": opb,
            "C": opc,
            "D": opd
        },
        # textual answer
        "answer": answer_to_text[answer][i],
        "answer_idx": answer,
    })  

import json
with open('../altro_dataset.jsonl', 'w') as outfile:
    for entry in formatted_question:
        json.dump(entry, outfile)
        outfile.write('\n')