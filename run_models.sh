declare -a models=(
    'HuggingFaceH4/zephyr-7b-beta'
    'meta-llama/Llama-2-7b-chat-hf'
    'mistralai/Mistral-7B-Instruct-v0.1'
)

# declare a string variable 
declare -a datasets=(
    'medqa_opt4'
    'medqa_opt5'
    'medmcqa_opt4'
)

rerank=true

datasets_string="${datasets[0]}"

debug=false

for dataset in "${datasets[@]:1}"; do
    datasets_string="$datasets_string,$dataset"
done

for model in "${models[@]}"; do
    python3 main.py --model $model --datasets $datasets_string --rerank $rerank --debug $debug
done
