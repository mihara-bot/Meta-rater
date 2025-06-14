# Example evaluation script using the lm-evaluation-harness framework

accelerate launch -m --main_process_port=8189 lm_eval --model hf \
    --model_args pretrained=MODEL_PATH \
    --trust_remote_code \
    --tasks arc_easy,arc_challenge,sciq,hellaswag,mmlu,social_iqa,winogrande,commonsense_qa,race,openbookqa,nq_open \
    --batch_size 16 \
    --write_out \
    --output_path output/ 