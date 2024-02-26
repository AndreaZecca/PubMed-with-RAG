# PubMed with RAG
This repository contains the code for the project "PubMed-with-RAG" for the course "DataMining" at the University of Bologna.

## Description
The goal of this project is to test several models on some QA tasks using the PubMed dataset. The models we tested are:
- Mistral (mistralai/Mistral-7B-Instruct-v0.1)
- Zephyr (HuggingFaceH4/zephyr-7b-beta)
- Llama (meta-llama/Llama-2-7b-chat-hf)

All the pipeline code is in the `main.py` file.

We used the `langchain` library to develop the pipeline.

We tested the model on four datasets:
1. MedQA-opt4
2. MedQA-opt5
3. MedMCQA-opt4
4. MMLU-opt4

## Workflow
All the code is in the `main.py` file. The pipeline is composed of `retriever`, `reranker` and `llm inference`. The `retriever` is the model that selects the most relevant documents from the PubMed dataset. The `reranker` is the model that selects the most relevant sentences from the documents selected by the `retriever`. The `llm inference` is the model that generates the answer to the question using the sentences selected by the `reranker`.

All the prompts used are in the `template` folder.

## Results
All the results are in the `results` folder.


## Authors
- [Andrea Zecca](https://github.com/AndreaZecca)
- [Antonio Lopez](https://github.com/elements72)
- [Matteo Vannucchi](https://github.com/MatteoVannucchi0)
- [Stefano Colamonaco](https://github.com/StefanoColamonaco)