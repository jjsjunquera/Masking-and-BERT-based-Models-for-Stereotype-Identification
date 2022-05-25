import torch
import statistics

class Head:
    def __init__(self, head):
        self.head=head

    def get_attention_per_tokens(self):

        sum_for_columns = torch.sum(self.head, dim=0) # size = [1, ncol]


        # sum_for_rows = torch.sum(self.head, dim=1) # # size = [nrow, 1]

        # print(sum_for_columns,sum_for_rows)
        return sum_for_columns.tolist()

    def get_attention_per_relations(self):
        # print(self.head.shape[0])
        d_result={}
        for f in range(self.head.shape[0]):
            for c in range(self.head.shape[1]):
                d_result[(f,c)] = self.head[f,c].tolist()
        return d_result


class Layer:
    def __init__(self, t_heads):
        # self.t_headsT = t_heads
        self.t_headsH = [Head(head) for head in t_heads]

    def get_attention_per_tokens(self):
        l_attentions_per_token = []
        # for each head, to take the list of attention per token
        for head in self.t_headsH:
            l_attention_per_token = head.get_attention_per_tokens()
            if len(l_attentions_per_token) == 0:
                l_attentions_per_token = [[attention_weight] for attention_weight in l_attention_per_token]
            else:
                for i in range(len(l_attention_per_token)):
                    l_attentions_per_token[i].append(l_attention_per_token[i])
        # for each token, to obtain the mean and standard deviation of its attentions in the heads
        l_mean, l_sd = [],[]
        for l_attention in l_attentions_per_token:
            # print(l_attention)
            mean = statistics.mean(l_attention)
            ds = 0.0
            if len(l_attention)>=2:
                ds = statistics.stdev(l_attention)
            l_mean.append(mean)
            l_sd.append(ds)

        return l_mean, l_sd

    def get_attention_per_relations(self):
        if len(self.t_headsH) >1:
            raise("What to do with more than one layer?")
        head = self.t_headsH[0]
        return head.get_attention_per_relations()



class InstanceWithAttentions():
    def __init__(self, code, text, tokens, layers, actual_label, predicted_label, prob_prediction):
        self.prob_prediction = prob_prediction
        self.predicted_label = predicted_label
        self.actual_label = actual_label
        # self.layers = layers
        self.layersL = [Layer(layer) for layer in layers]
        self.tokens = tokens
        self.text = text
        self.code = code

    def get_tokens_att(self, layer=11):
        if layer==None:
            raise
        return self.layersL[layer].get_attention_per_tokens()[0]

    def get_attention_per_relations(self, layer=11):
        if layer==None:
            raise
        d_relation_attention = self.layersL[layer].get_attention_per_relations()
        l_result = []
        for (t_f, t_c), att in d_relation_attention.items():
            token_f, token_c  = self.tokens[t_f],self.tokens[t_c]
            l_result.append((token_f, token_c, att))
        return l_result


class Analyser:
    def __init__(self,dirpath, file_pt_list, id, misclassified=False, all_from_id=False, all_instances=False):
        self.l_items = [ ]
        i = 0
        print("loading .pt files")
        for file_pt in file_pt_list:

            i+=1
            item = torch.load(dirpath+"/"+file_pt, map_location=torch.device('cpu'))
            instance = InstanceWithAttentions(code=item[0],
                                              text=item[1],
                                              tokens=item[2],
                                              actual_label=item[3].tolist(),
                                              predicted_label=item[4],
                                              prob_prediction=item[5],
                                              layers=item[6])
            if all_instances:
                self.l_items.append(instance)
            if all_from_id:
                if instance.actual_label == id:
                    self.l_items.append(instance)
            elif misclassified:
                if instance.predicted_label!=instance.actual_label:
                    if instance.predicted_label == id:
                        self.l_items.append(instance)
            else:
                if instance.predicted_label==instance.actual_label:
                    if instance.predicted_label == id:
                        self.l_items.append(instance)
                    # if len(self.l_items) == 20:
                    #     break
                # print()
            # else:
            #     remove(dirpath+"/"+file_pt)
        print("{} instances from .pt files".format(len(self.l_items)))


    def get_rigth_predictions(self, with_label=None, ordered = True):
        l = [inst for inst in self.l_items if inst.actual_label==inst.predicted_label and (with_label==None or with_label==inst.predicted_label)]
        return list(sorted(l, key=lambda  x: x.prob_prediction, reverse=True))
    
    def get_predictions(self, ordered = True):
        return list(sorted(self.l_items, key=lambda  x: x.prob_prediction, reverse=True))

    def get_top_more_attended_tokens(self, l_instances, n=10):
        d_token_attentions = {}
        d_token_examples = {}

        for inst in l_instances:
            att = inst.get_tokens_att(layer=11)
            tokens = inst.tokens
            for i in range(len(tokens)):
                token = tokens[i]
                if token not in d_token_attentions:
                    d_token_attentions[token] = []
                    d_token_examples[token] = []
                d_token_attentions[token].append(att[i]/len(l_instances))
                d_token_examples[token].append((att[i],inst))

        # for token, l_att in d_token_attentions.items():
        #     print(l_att)
        #     statistics.stdev(l_att)
        #
        result = [(token,
                   statistics.mean(l_att)*len(set([inst for _, inst in d_token_examples[token]])),
                   (statistics.mean(l_att),
                   0 if len(l_att) < 2 else statistics.stdev(l_att),
                   len(l_att)))
                  for token, l_att in d_token_attentions.items()]
        if n == -1:
            return list(sorted(result,key=lambda x:x[1], reverse=True)), d_token_examples
        return list(sorted(result,key=lambda x:x[1], reverse=True))[:n], d_token_examples

    def get_top_more_attended_relations(self, l_instances, n=10):
        l_result = []
        for inst in l_instances:
            inst_relations = inst.get_attention_per_relations()
            l_result.extend(inst_relations)
            break
        return list(sorted(l_result, key=lambda x:x[2], reverse=True))[:n]

