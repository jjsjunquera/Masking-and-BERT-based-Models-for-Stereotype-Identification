
from sklearn.feature_extraction.text import CountVectorizer
import re
import torch
from os import walk
from analyser import Analyser


def clusters(items_list, f_att=lambda d: None):
    dictionary = {}
    for item in items_list:
        att = f_att(item)
        if att not in dictionary.keys():
            dictionary[att] = []
        dictionary[att].append(item)
    return dictionary

from sklearn.model_selection import StratifiedKFold
def split(X,y,n_splits):
  skf = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
  for train_index, test_index in skf.split(X, y):
      X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
      y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
      return X_train, X_test, y_train, y_test

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

def visualize_confusion_matrix(y_pred_argmax, y_true):
    """

    :param y_pred_arg: This is an array with values that are 0 or 1
    :param y_true: This is an array with values that are 0 or 1
    :return:
    """

    cm = tf.math.confusion_matrix(y_true, y_pred_argmax).numpy()
    con_mat_df = pd.DataFrame(cm)

    print(classification_report(y_pred_argmax, y_true))

    sns.heatmap(con_mat_df, annot=True, fmt='g', cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

d_abrv_modelname={
                  # "beto":"beto/pytorch/",
                  "mbert":"bert-base-multilingual-cased",
                  "bert":"bert-base-uncased",
                  # "spanbert":"SpanBERT/spanbert-base-cased",
                  "xlmrobertabase": "xlm-roberta-base",
                  #"trans_xl": "transfo-xl-wt103"
                  # "xlm17":"xlm-mlm-17-1280",
                  # "xlm100":"xlm-mlm-100-1280",
                  # "xlmrobertalarge": "xlm-roberta-large"
                  }



def get_model_and_tokenizer(model_name):
    from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel, BertConfig, TransfoXLTokenizer, TransfoXLModel

    if model_name in ["mbert", "beto"]:
        #config = BertConfig.from_pretrained(d_abrv_modelname[model_name], output_attentions=True)
        # config = BertConfig.from_pretrained(d_abrv_modelname[model_name], output_attentions=True)
        # model =  BertModel.from_pretrained(d_abrv_modelname[model_name], output_hidden_states=True, return_dict=True, config=config)
        model =  BertModel.from_pretrained(d_abrv_modelname[model_name], output_hidden_states=True, return_dict=True, output_attentions=True)
        # model.config.num_attention_heads = 3
        # model.config.num_hidden_layers = 6
        tokenizer = BertTokenizer.from_pretrained(d_abrv_modelname[model_name])
    elif model_name == "trans_xl":
        model = TransfoXLModel.from_pretrained(d_abrv_modelname[model_name], output_hidden_states=True, return_dict=True, output_attentions=True)
        tokenizer = TransfoXLTokenizer.from_pretrained(d_abrv_modelname[model_name])
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # tokenizer.pad_token = tokenizer.eos_token
    else:
        model =  AutoModel.from_pretrained(d_abrv_modelname[model_name], output_hidden_states=True,return_dict=True,output_attentions=True)
        tokenizer = AutoTokenizer.from_pretrained(d_abrv_modelname[model_name])
    return model, tokenizer

def tokenizer_for_alpha(x):
    cv = CountVectorizer(lowercase=True)
    token_pattern = re.compile(cv.token_pattern)
    tokenizer = token_pattern.findall
    l_token = tokenizer(x.lower())
    l_result = []
    for token in l_token:
        if token.isalpha():
            l_result.append(token)
    return l_result

def save_attentions(attention, tokens):

    tensor = attention
    torch.save((tensor,tokens), 'file.pt')
    tensor1 = torch.load('file.pt')
    print(type(tensor1[1]))

# def save_layer(l_heads):


def save_matrices(inputs, tokenizer, d_text_code,aux_texts, layers, real_tag, predicted_tag, prob, i_or_ii):
    # layers: tuple of the 12 layers
    # print("inputs.shape")
    # print(inputs.shape)
    # print("layers")
    # print(layers[0][0].shape)
    # print(layers[0][12].shape)
    # raise Exception
    for i in range(inputs.shape[0]):
        input_id_list = inputs[i].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)
        text = aux_texts[i]
        code = d_text_code[text]
        # print(code, aux_texts[i], tokens)
        # for each layer I take the 12 heads of the instance i
        # layers_i = [layer[i].mean(dim=0,keepdim=True) for layer in layers]
        torch.save((code,text,tokens,real_tag[i],predicted_tag[i], prob[i], layers[i]), '/home/jjsjunquera/Stereotype/attentions_{0}/attentions_of_{1}.pt'.format(i_or_ii,code))

def for_saving(inputs, tokenizer, d_text_code,aux_texts, layers, real_tag, predicted_tag, prob, i_or_ii):
    result = []
    for i in range(inputs.shape[0]):
        input_id_list = inputs[i].tolist()
        tokens = tokenizer.convert_ids_to_tokens(input_id_list)
        text = aux_texts[i]
        code = d_text_code[text]
        # print(code, aux_texts[i], tokens)
        # for each layer I take the 12 heads of the instance i
        layers_i = [layer[i].mean(dim=0,keepdim=True) for layer in layers]
        result.append((code,text,tokens,real_tag[i],predicted_tag[i], prob[i], layers_i))
    return result

def save_matrices_from_list(l):
    for t in l:
        torch.save(t, '/content/drive/MyDrive/Stereotype/attentionsI/attentions_of_{0}.pt'.format(t[0]))

def load_att(id, task, misclassified=False, all_from_id=False, all_instances=False):
    x = [Analyser(dirpath,filenames, id, misclassified=misclassified, all_from_id=all_from_id, all_instances=all_instances) for (dirpath, dirnames, filenames) in walk("C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\\attentions_"+ task)]
    return x[0]

d_class_id = {"Stereotype":1, "Non-stereotype":0, "Victim":0, "Threat":1, "Left":0, "Main":1, "Rigth":2, "Truthful":1, "Deception":0}

def top_tokens(task_name, label, n=100, printing=False, misclassified=False, all=False):
    id = None if label==None else d_class_id[label]
    analyser = load_att(id, task_name, misclassified=misclassified, all_from_id=all,all_instances=label==None)
    # t_att, examples = analyser.get_top_more_attended_tokens(analyser.get_rigth_predictions(with_label=id),n=n)
    t_att, examples = analyser.get_top_more_attended_tokens(analyser.get_predictions(),n=n)

    if printing:
        for t, att in t_att:
            print(t, round(att,4))
            l_examples_with_t = [(round(attr,4),inst.text) for attr, inst in examples[t]]
            # print(t,len(l_examples_with_t),l_examples_with_t[:5])
    return t_att, examples


def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
        ("ñ", "n")
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s
