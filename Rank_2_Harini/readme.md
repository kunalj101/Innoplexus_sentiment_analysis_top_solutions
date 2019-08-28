I ran it on gpu. if you run on gpu it will take arround 8 hrs 
Please make code folder as the working directoy and run then following in terminal
After these final output will be created with name sub_all_avg.csv in code/output_final/sub_all_avg.csv
All the codes are in python3 


cd codes
pip install -r ../requirements.txt
python Data_prep.py

python train_k_fold_cross_val.py --dataset innoplex_sent --model_name bert_spc --learning_rate 1e-5 --num_epoch 7 --batch_size 16 --max_seq_len 150 --cross_val_fold 5 --polarities_dim 3 --hops 3 --give_weights 1 --assign_weights 0.0012,0.00026,0.0016 --seed 9


python train_k_fold_cross_val.py --dataset innoplex_sent --model_name bert_spc --learning_rate 1e-5 --num_epoch 7 --batch_size 16 --max_seq_len 150 --cross_val_fold 5 --polarities_dim 3 --hops 3 --give_weights 1 --assign_weights 0.0012,0.00026,0.0016 --seed 7

python train_k_fold_cross_val.py --dataset innoplex_sent --model_name bert_spc --learning_rate 1e-5 --num_epoch 7 --batch_size 16 --max_seq_len 150 --cross_val_fold 5 --polarities_dim 3 --hops 3 --give_weights 1 --assign_weights 0.0012,0.00026,0.0016 --seed 6

python train_k_fold_cross_val.py --dataset innoplex_sent --model_name aen_bert --learning_rate 1e-5 --num_epoch 6 --batch_size 16 --max_seq_len 150 --cross_val_fold 5 --polarities_dim 3 --hops 3 --give_weights 1 --assign_weights 0.0012,0.00026,0.0016  --seed 9

python ensemble.py


