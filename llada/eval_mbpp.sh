# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/ours/mbpp_bs1

log_folder=logs/ours/mbpp_bs1
timestamp=$(date +"%Y%m%d_%H%M%S")


task=mbpp
length=512
window_length=16
num_fewshot=3
steps=$((length / block_length))
model='GSAI-ML/LLaDA-8B-Instruct'
# model='GSAI-ML/LLaDA-1.5'
threshold=0.9
gamma=0.9
track_num=1
block_caching=True

CUDA_VISIBLE_DEVICES=0 accelerate launch eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist --batch_size 1 \
--model_args model_path=${model},gen_length=${length},steps=${steps},block_length=${block_length},threshold=${threshold},gamma=${gamma},track_num=${track_num},block_caching=${block_caching},show_speed=True \
--output_path evals_results/ours/mbpp-ns0-${length} --log_samples \
> ${log_folder}/${task}_baseline_len${length}_${timestamp}.log 

