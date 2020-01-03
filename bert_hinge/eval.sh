source activate transformers

export TASK_NAME=faq

python faq.py \
    --model_type bert \
    --model_name_or_path hinge_models/ \
    --config_name hinge_models/ \
    --tokenizer_name hinge_models/ \
    --task_name $TASK_NAME \
    --do_eval \
    --do_lower_case \
    --data_dir preprocessed.csv \
    --output_dir hinge_models \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32   
