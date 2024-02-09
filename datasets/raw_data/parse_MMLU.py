import pandas as pd

DATA_PATH = 'mmlu_medical 1.csv'

df = pd.read_csv(DATA_PATH)

# get questions
questions = df['question'].values

# get options
options_a = df['opa'].values
options_b = df['opb'].values
options_c = df['opc'].values
options_d = df['opd'].values

# get correct answer
answers = df['cop'].values

answer_to_text = {
    "A": options_a,
    "B": options_b,
    "C": options_c,
    "D": options_d
}


formatted_question = []

for i, (question, opa, opb, opc, opd, answer) in enumerate(zip(questions, options_a, options_b, options_c, options_d, answers)):
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
with open('../mmlu_opt4.jsonl', 'w') as outfile:
    for entry in formatted_question:
        json.dump(entry, outfile)
        outfile.write('\n')