#!/bin/bash
echo "Starting the experiments"
echo 
echo
echo
echo "230  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer rmsprop --batchsize 16 --folds 10 --f_load_source read_rev_doctor --f_load_target read_rev_restaurant"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer rmsprop --batchsize 16 --folds 10 --f_load_source read_rev_doctor --f_load_target read_rev_restaurant --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_rmsprop\%batchsize_16\%folds_10\%f_load_source_read_rev_doctor\%f_load_target_read_rev_restaurant.out"
echo 
echo
echo
echo "208  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "mbert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_doctor --f_load_target read_rev_hotel"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "mbert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_doctor --f_load_target read_rev_hotel --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"mbert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_rev_doctor\%f_load_target_read_rev_hotel.out"
echo 
echo
echo
echo
echo
echo "14  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_ab --f_load_target read_contr_bf"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_ab --f_load_target read_contr_bf --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_contr_ab\%f_load_target_read_contr_bf.out"
echo 
echo
echo
echo "38  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_ab --f_load_target read_contr_dp"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_ab --f_load_target read_contr_dp --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_contr_ab\%f_load_target_read_contr_dp.out"
echo 
echo
echo
echo
echo
echo "62  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_bf --f_load_target read_contr_ab"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_bf --f_load_target read_contr_ab --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_contr_bf\%f_load_target_read_contr_ab.out"
echo 
echo
echo
echo "86  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_bf --f_load_target read_contr_dp"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_bf --f_load_target read_contr_dp --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_contr_bf\%f_load_target_read_contr_dp.out"
echo 
echo
echo
echo "113  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer rmsprop --batchsize 16 --folds 10 --f_load_source read_contr_dp --f_load_target read_contr_ab"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer rmsprop --batchsize 16 --folds 10 --f_load_source read_contr_dp --f_load_target read_contr_ab --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_rmsprop\%batchsize_16\%folds_10\%f_load_source_read_contr_dp\%f_load_target_read_contr_ab.out"
echo 
echo
echo
echo "134  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_dp --f_load_target read_contr_bf"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_contr_dp --f_load_target read_contr_bf --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_contr_dp\%f_load_target_read_contr_bf.out"
echo 
echo
echo
echo
echo "161  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer rmsprop --batchsize 16 --folds 10 --f_load_source read_rev_hotel --f_load_target read_rev_doctor"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer rmsprop --batchsize 16 --folds 10 --f_load_source read_rev_hotel --f_load_target read_rev_doctor --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_rmsprop\%batchsize_16\%folds_10\%f_load_source_read_rev_hotel\%f_load_target_read_rev_doctor.out"
echo 
echo
echo
echo "182  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_hotel --f_load_target read_rev_restaurant"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_hotel --f_load_target read_rev_restaurant --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_rev_hotel\%f_load_target_read_rev_restaurant.out"
echo 
echo
echo
echo
echo "254  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_restaurant --f_load_target read_rev_hotel"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_restaurant --f_load_target read_rev_hotel --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_rev_restaurant\%f_load_target_read_rev_hotel.out"
echo 
echo
echo
echo "278  /home/jjsjunquera/Stereotype/main.py   --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_restaurant --f_load_target read_rev_doctor"
python  /home/jjsjunquera/Stereotype/main.py  --task "t_d" --model "bert" --maxlenght 172 --epochs 10 --learning 3e-05 --optimizer adam --batchsize 16 --folds 10 --f_load_source read_rev_restaurant --f_load_target read_rev_doctor --output "/home/jjsjunquera/Stereotype/output/deception/\%task_"t_d"\%model_"bert"\%maxlenght_172\%epochs_10\%learning_3e-05\%optimizer_adam\%batchsize_16\%folds_10\%f_load_source_read_rev_restaurant\%f_load_target_read_rev_doctor.out"
echo 
echo
echo
echo "Finishing"