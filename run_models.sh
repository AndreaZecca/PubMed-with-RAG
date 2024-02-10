declare -a models=(
    'meta-llama/Llama-2-7b-chat-hf'
    'HuggingFaceH4/zephyr-7b-beta'
    'mistralai/Mistral-7B-Instruct-v0.1'
)

declare -a datasets=(
    'medqa_opt4'
    'medqa_opt5'
    'medmcqa_opt4'
    'mmlu_opt4'
)

debug=false
rag=true
collection=medwiki_artificial

declare -a reranks=(
    'true'
    'false'
)

datasets_string="${datasets[0]}"
for dataset in "${datasets[@]:1}"; do
    datasets_string="$datasets_string,$dataset"
done

for model in "${models[@]}"; do
    for rerank in "${reranks[@]}"; do
        python3 main.py --model $model --datasets $datasets_string --rerank $rerank --debug $debug --rag $rag --collection $collection
    done
done
