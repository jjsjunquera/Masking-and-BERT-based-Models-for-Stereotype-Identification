from os import walk
from os import remove
import statistics
import torch

from execution import execute
from load_data import get_corpus
from auxiliarFunction import TaskStereotype
from setting import d_mode
from visualize import create_latex
from utils import top_tokens, d_class_id, load_att







def top_features_tru_positive():
    analyser = load_att(id, task="pro_anti", all_instances=True)
    beto_tp = [inst for inst in analyser.l_items if inst.predicted_label==inst.actual_label]
    print(analyser.get_top_more_attended_tokens(l_instances=beto_tp,n=10)[0])

# top_features_tru_positive()

def pruebas(inst_i,layer_i,header_i):
    task, label = "pro_anti", "Threat"
    # task, label = "pro_anti", "Victim"
    # task, label = "in_taxonomy", "Stereotype"
    # task, label = "in_taxonomy", "Non-stereotype"
    use_masking = True
    corpus = None
    if use_masking:
        corpus = masking(task_name=task)

    def given_class(label):
        id = d_class_id[label]
        analyser = load_att(id, task)
        right_pred = analyser.get_rigth_predictions(with_label=id)
        l_visualize = []

        for inst_analyser in right_pred[:100]:
            if (not use_masking) or id == corpus.d_code_maskresults[inst_analyser.code]["predicted"]:
                tokens = inst_analyser.tokens
                masked = ""
                if use_masking:
                    masked = corpus.d_code_maskresults[inst_analyser.code]["masked_text"]
                tokens_attention = inst_analyser.layersL[layer_i].t_headsH[0].get_attention_per_tokens()
                t_aux, att_aux = [], []
                for t in range(len(tokens)):
                    if tokens[t] in ["[PAD]" ,  "[SEP]", "[CLS]"]:
                        continue
                    else:
                        t_aux.append(tokens[t])
                        att_aux.append(tokens_attention[t])
                tokens = t_aux
                tokens_attention = att_aux
                # l_visualize.append((inst_analyser.code,inst_analyser.prob_prediction, tokens, tokens_attention, masked))
                l_visualize.append((inst_analyser.code+" MEAN",inst_analyser.prob_prediction, tokens, tokens_attention, masked))
        create_latex(l_visualize, label)
        print(len(l_visualize))



    # create latex file with colored examples
    given_class(label)







def true_pos_tokens(task_name):
    ## More attended tokens
    n = -1
    aux2, _ = top_tokens(task_name=task_name, label=None, n=n, printing=False)
    return [token for token,_,_ in aux2], \
           [], \
           [token for token,_,_ in aux2]

    if task_name=="in_taxonomy":
        top_s, examples_top_s = top_tokens(task_name="in_taxonomy", label="Stereotype", n=n, printing=False)
        top_n, examples_top_n = top_tokens(task_name="in_taxonomy", label="Non-stereotype", n=n, printing=False)
        top_s_misclassified, examples_top_s_misclassified = top_tokens(task_name="in_taxonomy", label="Stereotype", n=n, printing=False, misclassified=True)
        top_n_misclassified, examples_top_n_misclassified = top_tokens(task_name="in_taxonomy", label="Non-stereotype", n=n, printing=False, misclassified=True)

        top_only_s = [(token, attention, statis) for (token, attention, statis) in top_s if token not in [tok for tok,_ in top_n]]
        top_only_n = [(token, attention, statis) for (token, attention, statis) in top_n if token not in [tok for tok,_ in top_s]]

        top_only_s_minus_fp_s = [(token, attention, statis) for (token, attention, statis) in top_only_s if token not in [tok for tok,_ in top_s_misclassified]]
        top_only_n_minus_fp_n = [(token, attention, statis) for (token, attention, statis) in top_only_n if token not in [tok for tok,_ in top_n_misclassified]]

        print(len(top_only_s_minus_fp_s),len(top_only_n_minus_fp_n))

        # return [token for token,_,_ in top_only_v_minus_fp_v], \
        #        [token for token,_,_ in top_only_t_minus_fp_t], \
        #        [token for token,_,_ in top_v]+[token for token,_,_ in top_t]

        # return [token for token,_,_ in top_only_v], \
        #            [token for token,_,_ in top_only_t], \
        #            [token for token,_,_ in top_v]+[token for token,_,_ in top_t]

        # aux = list(sorted(top_v+top_t,key=lambda x:x[1],reverse=True))
        aux2, _ = top_tokens(task_name="pro_anti", label=None, n=n, printing=False)
        return [token for token,_,_ in aux2], \
               [], \
               [token for token,_,_ in aux2]
    else:
        top_v, examples_top_v = top_tokens(task_name="pro_anti", label="Victim", n=n, printing=False)
        top_t, examples_top_t = top_tokens(task_name="pro_anti", label="Threat", n=n, printing=False)
        top_v_misclassified, examples_top_v_misclassified = top_tokens(task_name="pro_anti", label="Victim", n=n, printing=False, misclassified=True)
        top_t_misclassified, examples_top_t_misclassified = top_tokens(task_name="pro_anti", label="Threat", n=n, printing=False, misclassified=True)

        top_only_v = [(token, attention, statis) for (token, attention, statis) in top_v if token not in [tok for tok,_,_ in top_t]]
        top_only_t = [(token, attention, statis) for (token, attention, statis) in top_t if token not in [tok for tok,_,_ in top_v]]

        top_only_v_minus_fp_v = [(token, attention, statis) for (token, attention, statis) in top_only_v if token not in [tok for tok,_,_ in top_v_misclassified]]
        top_only_t_minus_fp_t = [(token, attention, statis) for (token, attention, statis) in top_only_t if token not in [tok for tok,_,_ in top_t_misclassified]]

        print(len(top_only_v_minus_fp_v),len(top_only_t_minus_fp_t))

        # return [token for token,_,_ in top_only_v_minus_fp_v], \
        #        [token for token,_,_ in top_only_t_minus_fp_t], \
        #        [token for token,_,_ in top_v]+[token for token,_,_ in top_t]

        # return [token for token,_,_ in top_only_v], \
        #            [token for token,_,_ in top_only_t], \
        #            [token for token,_,_ in top_v]+[token for token,_,_ in top_t]

        # aux = list(sorted(top_v+top_t,key=lambda x:x[1],reverse=True))
        aux2, _ = top_tokens(task_name="pro_anti", label=None, n=n, printing=False)
        return [token for token,_,_ in aux2], \
               [], \
               [token for token,_,_ in aux2]


def creating_file_token_statistic():
    from nltk.corpus import stopwords
    for name_file in ("Stereotype", "Non-stereotype", "Victim", "Threat"):
        task_name =  "in_taxonomy" if name_file in ("Stereotype", "Non-stereotype") else "pro_anti"
        tokens, d_token_inst = top_tokens(task_name=task_name, label=name_file, n=-1, printing=False, all=True)
        f = open(name_file+".txt", "w")
        f.write("token & normalized attention & mean & stdev & num. instances & num. occurrences (used for mean and stdev)")
        f.write("\n")
        for token, att, statis in tokens:
            if token not in stopwords.words('spanish'):
                continue
            examples = list(set([ex for _,ex in d_token_inst[token]]))
            f.write("{} & {} & {} & {} & {} & {}".format(token, round(att,4), round(statis[0],4), round(statis[1],4), len(examples) , statis[2]))
            f.write("\n")
        f.close()
        
def intersections_models():
    d_task_corpus = {}
    for task_name in ("in_taxonomy", "pro_anti"):
        d_task_corpus[task_name] = masking(task_name=task_name).d_code_maskresults
    for task_name in d_task_corpus:
        lr_results = d_task_corpus[task_name]
        # corpus.d_code_maskresults[l_code[i]] = {"masked_text": x[i],
        #                                       "actual": y[i],
        #                                       "predicted": y_pred[i]}
        analyser = load_att(id, task_name, all_instances=True)
        print(task_name)
        print(len(lr_results))
        print(len(analyser.l_items))
        total = [inst.code for inst in analyser.l_items if inst.code in lr_results]
        print("total", len(total))
        lr_miscl = [code for code, d_res in lr_results.items() if d_res["actual"]!=d_res["predicted"]]
        print("misclassified by LR", len(lr_miscl))
        beto_miscl = [inst.code for inst in analyser.l_items if inst.predicted_label!=inst.actual_label]
        print("misclassified by BETO", len(beto_miscl))
        inter_miscl = len(set(beto_miscl).intersection(lr_miscl))
        print("misclassified by both", inter_miscl, "\n      {}% from LR and \n      {}% from BETO misclassifications".format(round(inter_miscl*100/len(lr_miscl),2),
                                                                                                              round(inter_miscl*100/len(beto_miscl),2)))

        inst_intesrection = [inst for inst in analyser.l_items if inst.code in set(beto_miscl).intersection(lr_miscl)]
        prob_int = [inst.prob_prediction for inst in inst_intesrection]
        print("prediction prob of misclassified by BETO (intersection)", round(statistics.mean(prob_int),2),
              "\n  stdev", round(statistics.stdev(prob_int),2),
              "\n  min", round(min(prob_int),2),
              "\n  max", round(max(prob_int),2))
        print("more attended tokens")
        for t in analyser.get_top_more_attended_tokens(inst_intesrection)[0]:
            print("   ",t)

        not_inst_intesrection = [inst for inst in analyser.l_items if inst.code in set(beto_miscl).difference(lr_miscl)]
        prob_not_int = [inst.prob_prediction for inst in not_inst_intesrection]
        print("prediction prob of misclassified ONLY by BETO ", round(statistics.mean(prob_not_int),2),
              "\n  stdev", round(statistics.stdev(prob_not_int),2),
              "\n  min", round(min(prob_not_int),2),
              "\n  max", round(max(prob_not_int),2))
        print("more attended tokens")
        for t in analyser.get_top_more_attended_tokens(not_inst_intesrection)[0]:
            print("   ",t)

def relations_attention():
    for task_name in ("in_taxonomy",):
        analyser = load_att(id, task_name, all_instances=True)
        l_results = analyser.get_top_more_attended_relations(analyser.get_rigth_predictions())
        for rel in l_results[:10]:
            print(rel[0], rel[1], rel[2])

def masking(task_name):
    task = TaskStereotype(task_name)
    analyser = None
    if d_mode["attention"]:
        analyser = load_att(id, task=task_name, all_instances=True)
    corpus = get_corpus(task, True, use_lr=True,attention=true_pos_tokens(task_name), analyser=analyser)
    l_text, l_label, l_code = corpus.get_texts_and_labels(f_get_tax=corpus.get_tax1)
    return corpus
    # print(list(corpus.d_code_maskresults.items())[0])

# masking(task_name="pro_anti")
# masking(task_name="in_taxonomy")
execute(task_name="t_f")
# relations_attention()
# Are the heads of a layer showing different attention weights to the same token?
# analyser = load_att()

# intersections_models()
# creating_file_token_statistic()
# pruebas(0,11,0)
# masking(task_name="in_taxonomy")
# true_pos_tokens()

