import argparse
import sys

from data_deception import d_name_f
from execution import execute, execute2
from utils import  d_abrv_modelname
import os



dp=0.3
batchsize=32
lr=3e-5#0.01

def check_params(args=None):
    parse = argparse.ArgumentParser(description='Method for automaticaly stereotype detection')
    parse.add_argument('-t', '--task', help='Task name', required=True, choices=["in_taxonomy", "pro_anti", "hyp_nohyp", "l_main_r","t_d"])
    parse.add_argument('-o', '--output', help='File path for writting the results and metrics', required=False, default="/home/jjsjunquera/Stereotype/output/x.out")
    parse.add_argument('-d', '--dropout', help='Dropout values used to reduce the overfitting ',  required=False, type=float, default=dp)
    parse.add_argument('-ml', '--maxlenght', help='Max length of the sequences used for training', required=False, type=int, default=128)
    parse.add_argument('-p', '--epochs', help='Number of epoch used in the training phase', required=False, type=int, default=5)
    parse.add_argument('-b', '--batchsize', help='Batch size used in the trainiing process', required=False, type=int, default=batchsize)
    parse.add_argument('-z', '--optimizer', help='Method used for parameters optimizations', required=False, type=str, default="rmsprop")#
    parse.add_argument('-r', '--learning', help='Value for the larning rate in the optimizer', required=False, type=float, default=lr)
    parse.add_argument('-md', '--model', help='Model', required=True, choices=list(d_abrv_modelname.keys()))
    parse.add_argument('-k', '--folds', help='Number of partitions in the cross-validation methods', required=False, type=int, default=5)
    parse.add_argument('-source', '--f_load_source', help='Function to load the source domain', required=False, choices=list(d_name_f.keys()))
    parse.add_argument('-target', '--f_load_target', help='Function to load the target domain', required=False, choices=list(d_name_f.keys()))
    #  f_load_source, f_load_target,
    results = parse.parse_args(args)
    return results


if __name__ == '__main__':
    params=check_params(sys.argv[1:])
    optimizer=params.optimizer
    lr=params.learning
    task =params.task
    output = params.output
    print(output)
    maxlenght = params.maxlenght
    epochs = params.epochs
    batchsize = params.batchsize
    model_name = params.model
    folds=params.folds
    f_load_source = params.f_load_source
    f_load_target = params.f_load_target
    print("Starting to execute the model!!!!")
    print("\n\n")
    print("MODEL", model_name)
    print("OPTIMIZER",optimizer)
    print("maxlenght",maxlenght)
    print("epochs", epochs)
    print("batchsize", batchsize)
    print("folds",folds)
    # if maxlenght == 183:
    #     print("maxlenght = 183")
    #     sys.exit(0)
    f = open(output, "w")
    if task == "t_d":
        execute2(task,f=f,model_name=model_name,max_length=maxlenght,opt=optimizer,learning_rate=lr,k=folds,
                 epochs=epochs,batch_size=batchsize, f_load_source=f_load_source,f_load_target=f_load_target)
    else:
        execute(task,model_name,maxlenght,optimizer,lr,batchsize,epochs,folds,f)
    f.close()

    print("Finishing")
    sys.exit(0)

