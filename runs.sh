#!/bin/bash
echo "Starting the experiments"
echo 
echo
echo "182  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_hotel --f_load_target read_rev_restaurant"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_hotel --f_load_target read_rev_restaurant --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_rev_hotel\%f_load_target_read_rev_restaurant.out"
echo 
echo
echo
echo "Finishing"