from data_deception import d_name_f
from utils import  d_abrv_modelname

f = open("runs.sh", "w")
f.write("#!/bin/bash")
f.write("\n"+"echo \"Starting the experiments\"")
i = 0
for task in ("t_d",):#,"in_taxonomy","pro_anti"):
    for f_load_source in d_name_f:
        for f_load_target in d_name_f:
          if ("_contr_" in f_load_source and "_contr_" in f_load_target) or ("_rev_" in f_load_source and "_rev_" in f_load_target):
            if f_load_source == f_load_target:
              continue
            for batchsize in (16,):
                for lr  in [3e-03,3e-04,3e-05,0.01]:
                    for optimizer in ["adam","rmsprop"]:
                        for maxlength in [172]:
                                for model in d_abrv_modelname:
                                    i += 1
                                    string = " --task \"{0}\" --model \"{1}\" --maxlenght {2} --epochs {3} --learning {4} --optimizer {5} --batchsize {6} --folds 10 --f_load_source {7} --f_load_target {8}".format(task,model,maxlength,10,lr,optimizer, batchsize, f_load_source, f_load_target)
                                    f.write("\n"+"echo ")
                                    f.write("\n"+"echo")
                                    f.write("\n"+"echo")
                                    f.write("\n"+"echo \"{0}  /home/jjsjunquera/Stereotype/main.py  {1}\"".format(i,string))
                                    f.write("\n"+"python  /home/jjsjunquera/Stereotype/main.py ")
                                    f.write(string+ " --output \"/home/jjsjunquera/Stereotype/output{0}/{1}.out\"".format("/deception" if task=="t_d" else "",string.replace(" --", "\%").replace(" ", "_")))
#
# echo "1 /home/jjsjunquera/Stereotype/%taskname_in_taxonomy%modelname_mbert%maxlength_64%epoch_5%lr_3e-05%optimizer_rmsprop%batchsize_32%folds_5"
# python  /home/jjsjunquera/Stereotype/main.py --task "in_taxonomy" --model "mbert" --maxlenght 64 --epochs 5 --learning 3e-05 --optimizer rmsprop --batchsize 32 --folds 5 --output "/home/jjsjunquera/Stereotype/output/ --task in_taxonomy --model mbert --maxlenght 64 --epochs 5 --learning 3e-05 --optimizer rmsprop --batchsize 32 --folds 5.out"
# echo "2 /home/jjsjunquera/Stereotype/%taskname_in_taxonomy%modelname_beto%maxlength_64%epoch_5%lr_3e-05%optimizer_rmsprop%batchsize_32%folds_5"
# python  /home/jjsjunquera/Stereotype/main.py --task "in_taxonomy" --model "beto" --maxlenght 64 --epochs 5 --learning 3e-05 --optimizer rmsprop --batchsize 32 --folds 5 --output "/home/jjsjunquera/Stereotype/output/ --task in_taxonomy --model beto --maxlenght 64 --epochs 5 --learning 3e-05 --optimizer rmsprop --batchsize 32 --folds 5.out"
f.write("\n"+"echo \"Finishing\"")
f.close()
