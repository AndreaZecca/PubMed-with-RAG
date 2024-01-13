declare -a models=(
    'HuggingFaceH4/zephyr-7b-beta'
    'meta-llama/Llama-2-7b-chat-hf'
    # 'mistralai/Mistral-7B-Instruct-v0.1'
)

# declare a string variable 
datasets="medqa_opt4,medqa_opt5,medmcqa_opt4"

for model in "${models[@]}"; do
    # passing the datasets as a 
    python3 main.py --model $model --datasets $datasets
done