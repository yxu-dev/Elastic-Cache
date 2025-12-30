# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=gsm8k
length=512
window_length=32
num_fewshot=5
steps=$((length / window_length))
# model="Dream-org/Dream-v0-Base-7B"
model="Dream-org/Dream-v0-Instruct-7B"
threshold=0.9
gamma=0.9
track_num=1
block_caching=True

accelerate launch eval.py --model dream \
    --model_args pretrained=${model},max_new_tokens=${length},diffusion_steps=${steps},add_bos_token=true,alg=confidence_threshold,threshold=${threshold},gamma=${gamma},window_length=${window_length},track_num=${track_num},block_caching=${block_caching},show_speed=True,escape_until=true \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --confirm_run_unsafe_code


# ## NOTICE: use postprocess for humaneval
# python postprocess_code.py {the samples_xxx.jsonl file under output_path}
