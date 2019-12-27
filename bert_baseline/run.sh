export TASK_NAME=faq

python faq_predict.py \
    --model_type bert \
    --model_name_or_path /share/data/lang/users/zeweichu/projs/qa-chinese/chinese_wwm_ext_pytorch \
    --config_name /share/data/lang/users/zeweichu/projs/qa-chinese/chinese_wwm_ext_pytorch \
    --tokenizer_name /share/data/lang/users/zeweichu/projs/qa-chinese/chinese_wwm_ext_pytorch \
    --task_name $TASK_NAME \
    --do_train \
    --do_eval \
    --do_lower_case \
    --data_dir ../../dataset/faq/lawzhidao_filter.csv \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=32   \
    --learning_rate 2e-5 
