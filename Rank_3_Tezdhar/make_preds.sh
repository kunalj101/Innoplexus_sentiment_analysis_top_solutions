#for i in 4 1 0; do
#python -m src.bert_finetuning --data_dir data/v2/fold_$i \
#                              --model_type xlnet \
#                              --model_name_or_path 'xlnet-base-cased' \
#                              --task_name inno \
#                              --output_dir models/xlnet_v2/fold_$i \
#                              --max_seq_length 150 \
#                              --do_train \
#                              --do_eval \
#                              --evaluate_during_training \
#                              --per_gpu_train_batch_size 32 \
#                              --per_gpu_eval_batch_size 64 \
#                              --gradient_accumulation_steps 1 \
#                              --learning_rate 8e-6 \
#                              --weight_decay 0 \
#                              --num_train_epochs 8 \
#                              --save_steps 132 \
#                              --logging_steps 132 \
#                              --overwrite_output_dir \
#                              --overwrite_cache \
#                              --warmup_steps 0 \
#                              --seed 786
#done
#for i in 0 1 2 3 4; do
#python -m src.bert_finetuning --data_dir data/v2/fold_$i \
#                              --model_type bert \
#                              --model_name_or_path 'bert-base-uncased' \
#                              --task_name inno \
#                              --output_dir models/bert_v2/fold_$i \
#                              --max_seq_length 180 \
#                              --do_train \
#                              --do_eval \
#                              --evaluate_during_training \
#                              --per_gpu_train_batch_size 32 \
#                              --per_gpu_eval_batch_size 64 \
#                              --gradient_accumulation_steps 1 \
#                              --learning_rate 1e-5 \
#                              --weight_decay 0 \
#                              --num_train_epochs 8 \
#                              --save_steps 132 \
#                              --logging_steps 132 \
#                              --overwrite_output_dir \
#                              --overwrite_cache \
#                              --warmup_steps 0 \
#                              --seed 786
#done
#for i in 0 1 2; do
#python -m src.bert_finetuning --data_dir data/v2/fold_$i \
#                              --model_type bert \
#                              --model_name_or_path 'bert-large-uncased' \
#                              --task_name inno \
#                              --output_dir models/bert_v3/fold_$i \
#                              --max_seq_length 150 \
#                              --do_train \
#                              --do_eval \
#                              --evaluate_during_training \
#                              --per_gpu_train_batch_size 8 \
#                              --per_gpu_eval_batch_size 64 \
#                              --gradient_accumulation_steps 4 \
#                              --learning_rate 1e-5 \
#                              --weight_decay 0 \
#                              --num_train_epochs 8 \
#                              --save_steps 132 \
#                              --logging_steps 132 \
#                              --overwrite_output_dir \
#                              --overwrite_cache \
#                              --warmup_steps 0 \
#                              --seed 786
#done
#for i in 0 1 2; do
#python -m src.bert_finetuning --data_dir data/v3/fold_$i \
#                              --model_type xlnet \
#                              --model_name_or_path 'xlnet-base-cased' \
#                              --task_name inno \
#                              --output_dir models/xlnet_v3/fold_$i \
#                              --max_seq_length 140 \
#                              --do_train \
#                              --do_eval \
#                              --evaluate_during_training \
#                              --per_gpu_train_batch_size 32 \
#                              --per_gpu_eval_batch_size 64 \
#                              --gradient_accumulation_steps 1 \
#                              --learning_rate 8e-6 \
#                              --weight_decay 0 \
#                              --num_train_epochs 8 \
#                              --save_steps 132 \
#                              --logging_steps 132 \
#                              --overwrite_output_dir \
#                              --overwrite_cache \
#                              --warmup_steps 0 \
#                              --seed 786
#done
for i in 0; do
python -m src.bert_finetuning --data_dir data/v3/fold_$i \
                              --model_type bert \
                              --model_name_or_path 'bert-large-uncased' \
                              --task_name inno \
                              --output_dir models/bert_v4/fold_$i \
                              --max_seq_length 150 \
                              --do_eval \
                              --evaluate_during_training \
                              --per_gpu_train_batch_size 8 \
                              --per_gpu_eval_batch_size 64 \
                              --gradient_accumulation_steps 4 \
                              --learning_rate 1e-5 \
                              --weight_decay 0 \
                              --num_train_epochs 8 \
                              --save_steps 132 \
                              --logging_steps 132 \
                              --overwrite_output_dir \
                              --overwrite_cache \
                              --warmup_steps 0 \
                              --seed 786
done
