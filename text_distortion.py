# coding=utf-8
import random
import statistics
from math import log
from setting import mask_frequents, to_mask_sw, d_mode

import nltk


from ch_family import ChFamilyMethods
from utils import normalize
from nltk.corpus import stopwords

class TextDistortion:
    """Based on Authorship Attribution Using Text Distortion by Stamatatos"""

    def __init__(self, list_frequents_words=[], fun=None, mask_frequents=True, from_BETO = False,vocab=None):
        """

        :type list_frequents_words: list[str]
        """
        self.mask_frequents = mask_frequents
        print("mask most frequents", self.mask_frequents)
        self.function = fun
        self.list_frequents_words = list_frequents_words
        self.vocab, self.rel_freq, self.analyser = vocab
        self.set_in_list = set([])
        if from_BETO:
            self.in_list_freq_words = self.in_list_freq_words_from_beto
        else:
            self.in_list_freq_words = self.in_list_freq_words_default

    def distort_document(self, l_tokens, func=None):
        """

        :param func: lambda
        :type l_tokens: list[str]
        """
        l_distorted = []
        for token in l_tokens:
            # if token not in self.list_frequents_words:
            #     continue
            l_distorted.append(self.distort_token(token, func))
        return l_distorted

    def distort_token(self, token, func=None):
        """

        :param func: lambda
        :type token: str
        """
        if func is None:
            func = self.function
        if func is None:
            return token
        # in the list of the k terms
        is_one_of_k = self.in_list_freq_words(token)
        # deciding if it will be masked or maintained intact
        to_mask = (self.mask_frequents and is_one_of_k) or (not self.mask_frequents and not is_one_of_k)
        if to_mask:
            return func(token)
        else:
            return token



    @staticmethod
    def dv_sa(token):
        """Distorted View with Single Asterisks (DV-SA)"""
        if ChFamilyMethods.is_word_fragment(token):
            return "*"
        elif ChFamilyMethods.has_number(token):
            return "#"
        else:
            # quitar el lanzado de excepcion, es solo para saber en que caso puede estar mezclado letra y numero
            # o no ser ninguno y ser poco frecuente

            return token  # "@"

    @staticmethod
    def dv_saa_num_wcard(token):
        """Distorted View with Single Asterisks (DV-SA), @, #, +"""

        if ChFamilyMethods.has_number(token):
            return "#"
        elif nltk.pos_tag([token])[0][1] == "CD":
            return "+"
        elif ChFamilyMethods.is_word_fragment(token):
            return "*"
        else:
            # quitar el lanzado de excepcion, es solo para saber en que caso puede estar mezclado letra y numero
            # o no ser ninguno y ser poco frecuente

            return "@"

    @staticmethod
    def dv_saa_num_eq_wcard(token):
        """Distorted View with Single Asterisks (DV-SA), @, #=+"""

        if ChFamilyMethods.has_number(token) or nltk.pos_tag([token])[0][1] == "CD":
            return "#"
        elif ChFamilyMethods.is_word_fragment(token):
            return "*"
        else:
            # quitar el lanzado de excepcion, es solo para saber en que caso puede estar mezclado letra y numero
            # o no ser ninguno y ser poco frecuente

            return "@"

    @staticmethod
    def dv_saa_wcard(token):
        """Distorted View with Single Asterisks (DV-SA), @, +"""

        if ChFamilyMethods.has_number(token):
            return token
        elif nltk.pos_tag([token])[0][1] == "CD":
            return "+"
        elif ChFamilyMethods.is_word_fragment(token):
            return "*"
        else:
            # quitar el lanzado de excepcion, es solo para saber en que caso puede estar mezclado letra y numero
            # o no ser ninguno y ser poco frecuente

            return "@"

    @staticmethod
    def dv_saa(token):
        """Distorted View with Single Asterisks (DV-SA), @"""

        if ChFamilyMethods.has_number(token):
            return "#"
        elif ChFamilyMethods.is_word_fragment(token):
            return "*"
        else:
            # quitar el lanzado de excepcion, es solo para saber en que caso puede estar mezclado letra y numero
            # o no ser ninguno y ser poco frecuente
            # print(token)
            return "@"

    @staticmethod
    def dv_maa_num_wcard(token):
        """Distorted View with Multiple Asterisks (DV-MA)"""
        if nltk.pos_tag([token])[0][1] == "CD":  # por aqui entra si es cardinal o digitos
            if ChFamilyMethods.has_number(token):
                return '#' * len(token)
            return '+' * len(token)
        if ChFamilyMethods.is_word_fragment(token):
            return '*' * len(token)
        return '@'



    @staticmethod
    def dv_maa(token):
        """Distorted View with Multiple Asterisks (DV-MA)"""
        distorted = ""
        for char in token:
            if ChFamilyMethods.is_letter(char):
                distorted += "*"
            elif ChFamilyMethods.has_number(char):
                distorted += "#"
            else:
                distorted += "@"
        return distorted

    @staticmethod
    def dv_ma(token):
        """Distorted View with Multiple Asterisks (DV-MA)"""
        distorted = ""
        for char in token:
            if ChFamilyMethods.is_letter(char):
                distorted += "*"
            elif ChFamilyMethods.has_number(char):
                distorted += "#"
            else:
                distorted += char
        return distorted

    @staticmethod
    def dv_ex(token):
        """Distorted View - Exterior Characters (DV-EX)"""
        distorted = ''
        for i in range(len(token)):
            if i != 0 and i != len(token) - 1 and ChFamilyMethods.is_letter(token[i]):
                distorted += "*"
            elif ChFamilyMethods.has_number(token[i]):
                distorted += "#"
            else:
                distorted += token[i]

        return distorted

    @staticmethod
    def dv_ex_num_wcard(token):
        """Distorted View - Exterior Characters (DV-EX)"""
        if nltk.pos_tag([token])[0][1] == "CD":  # por aqui entra si es cardinal o digitos
            if ChFamilyMethods.has_number(token):
                return '#' * len(token)
            return '+' * len(token)
        if ChFamilyMethods.is_word_fragment(token):
            return token[0]+'*' * (len(token)-2)+token[-1]
        return '@'

    @staticmethod
    def dv_exn(token):
        """Distorted View - Exterior Characters (DV-EX)"""
        distorted = ''
        for i in range(len(token)):
            if i != 0 and i != len(token) - 1:
                if ChFamilyMethods.is_letter(token[i]):
                    distorted += "*"
                elif ChFamilyMethods.has_number(token[i]):
                    distorted += "#"
            else:
                distorted += token[i]

        return distorted

    @staticmethod
    def dv_l2n(token):
        """Distorted View - Last 2 Characters (DV-L2)"""
        distorted = ""
        for char in token[:-2]:
            if ChFamilyMethods.is_letter(char):
                distorted += "*"
            elif ChFamilyMethods.has_number(char):
                distorted += "#"
            else:
                distorted += char
        distorted += token[-2:]
        return distorted

    @staticmethod
    def dv_l2(token):
        """Distorted View - Last 2 Characters (DV-L2)"""
        distorted = ""
        for i in range(len(token)):
            char = token[i]
            if i < len(token) - 2:
                if ChFamilyMethods.is_letter(char):
                    distorted += "*"
            if ChFamilyMethods.has_number(char):
                distorted += "#"
            else:
                distorted += char

        return distorted

    @staticmethod
    def dv_l2_num_wcard(token):
        """Distorted View - Last 2 Characters (DV-L2)"""
        if nltk.pos_tag([token])[0][1] == "CD":  # por aqui entra si es cardinal o digitos
            if ChFamilyMethods.has_number(token):
                return '#' * len(token)
            return '+' * len(token)
        if ChFamilyMethods.is_word_fragment(token):
            return '*' * (len(token)-2)+token[-2:]
        return '@'


    @staticmethod
    def dv_nu(token):
        if ChFamilyMethods.has_number(token):
            return "#"
        else:
            return ""
        return token

    @staticmethod
    def dv_ra(token):
        arr = [TextDistortion.dv_ma, TextDistortion.dv_saa, lambda t: t]
        return arr[random.randint(0, len(arr) - 1)](token)

    class SeekFrequentsWords:
        def __init__(self, list_of_tokenized_docs):
            """

            :type list_of_tokenized_docs: list[list[str]]
            """
            self.list_of_tokenizedDocs = list_of_tokenized_docs
            self.dict_word_frequency = {}
            self.dict_word_cant_docs = {}  # cantidad de docs en los que está el término


        def calculate_frequencies(self):
            for l_words in self.list_of_tokenizedDocs:
                for word in l_words:
                    if word in self.dict_word_frequency:
                        self.dict_word_frequency[word] += 1
                    else:
                        self.dict_word_frequency[word] = 1

                for word in set(l_words):
                    if word in self.dict_word_cant_docs:
                        self.dict_word_cant_docs[word] += 1
                    else:
                        self.dict_word_cant_docs[word] = 1

        def words_with_at_least(self, frequency):
            return [word for word, f, in self.dict_word_frequency.items() if f >= frequency]

        def most_frequents(self, k, ignore=False, present_in=None):
            # type: (int) -> list[str]
            if len(self.dict_word_frequency) == 0:
                self.calculate_frequencies()
            # if k == 0:
            #     return []
            list_ordered_w_and_f = sorted(self.dict_word_frequency.items(), key=lambda x: x[1], reverse=True)
            if k == -1:  # todos
                return [w for w, _ in list_ordered_w_and_f]
            # fw = [term for term in self.fw if term in self.dict_word_frequency.keys()]

            list_ordered_w = []
            for w, _ in list_ordered_w_and_f:
                if not ignore:
                    if k == 0:
                        break
                    # si debe estar en present_in
                    if (present_in is not None) and (w not in present_in):
                        continue

                    list_ordered_w.append(w)
                else:
                    if k > 0:
                        continue
                    list_ordered_w.append(w)
                k -= 1
            return list_ordered_w

            return [term for term in self.fw if term in self.dict_word_frequency.keys()]



        @staticmethod
        def FCE(seekFrequentsWordsSource, seekFrequentsWordsTarget, alpha=0.0001, beta=0.0001):
            fce = {}
            from nltk.corpus import stopwords
            sw = stopwords.words('spanish')
            voc_len_source, voc_len_tagert = len(seekFrequentsWordsSource.dict_word_frequency), len(
                seekFrequentsWordsTarget.dict_word_frequency)
            voc = set(
                list(seekFrequentsWordsSource.dict_word_frequency.keys()) + list(seekFrequentsWordsTarget.dict_word_frequency.keys()))
            dict_borrar = {}

            for w in voc:
                frec_S = 0
                if w in seekFrequentsWordsSource.dict_word_frequency:
                    # frec_S = seekFrequentsWordsSource.dict_word_frequency[w]
                    frec_S = seekFrequentsWordsSource.dict_word_cant_docs[w]
                frec_T = 0
                if w in seekFrequentsWordsTarget.dict_word_frequency:
                    # frec_T = seekFrequentsWordsTarget.dict_word_frequency[w]
                    frec_T = seekFrequentsWordsTarget.dict_word_cant_docs[w]

                probS = (frec_S + alpha) / (voc_len_source + 2 * alpha)
                probT = (frec_T + alpha) / (voc_len_tagert + 2 * alpha)

                fce[w] = log((probS * probT) / (abs(probS - probT) + beta), 2)
                dict_borrar[w] = (frec_S,frec_T,fce[w])

            if d_mode['fce']:
                list_ordered_w_and_f = sorted(fce.items(), key=lambda x: x[1], reverse=True)
            if d_mode['diference_per_label']:
                aux = sorted(dict_borrar.items(), key=lambda x: x[1][0]*x[1][1], reverse=True)
                aux = sorted(aux, key=lambda x: abs(x[1][0]-x[1][1]), reverse=False)
                # for w, (s,t,fce) in aux:
                #     if w not in stopwords.words('spanish') and s>30:
                #         print(w,s,t)
                list_ordered_w_and_f = sorted(dict_borrar.items(), key=lambda x: abs(x[1][0]-x[1][1]), reverse=True)
            # print("\nFCE ranking",)
            for i in range(len(list_ordered_w_and_f)):
                if i == 1000:
                    break
                if list_ordered_w_and_f[i][0] in dict_borrar:
                    dict_borrar[list_ordered_w_and_f[i][0]] = (dict_borrar[list_ordered_w_and_f[i][0]], i)
                    # print(list_ordered_w_and_f[i][0],(dict_borrar[list_ordered_w_and_f[i][0]], i))

            list_dict_ordered_w_and_f = sorted(dict_borrar.items(), key=lambda x: x[1][1], reverse=False)




            # d_class_w ={"Threat":[],"Victims":[]}
            # d_class_w ={"Stereotype":[],"Non-stereotype":[]}
            # for w, (s,t,fce) in list_ordered_w_and_f:
            #     if w not in sw:
            #         # d_class_w["Threat" if s>t else "Victims"].append(w)
            #         d_class_w["Stereotype" if s>t else "Non-stereotype"].append(w)
            # for k,list_w in d_class_w.items():
            #     print(k)
            #     for w in list_w[:30]:
            #         print(w)
            #     print()
            return list_ordered_w_and_f

    def in_list_freq_words_from_beto(self, token):

        result = False
        if token in self.vocab or normalize(token) in self.vocab:
            if self.in_list_freq_words_default(token) or self.in_list_freq_words_default(normalize(token)):
                result = True
                # if result and not token in self.rel_freq:
                #     print(token)
        else:
            cont_true = 0
            for feat in self.list_frequents_words:
                if feat.startswith("##") and len(feat)>3:
                    if (feat[2:] in token and not token.startswith(feat[2:])) or (feat[2:] in normalize(token) and not normalize(token).startswith(feat[2:])):
                        cont_true += 1
                        break
            for feat in self.list_frequents_words:
                if len(feat)>3 and (token.startswith(feat) or normalize(token).startswith(feat)):
                    cont_true += 1
                    break
            if cont_true >1:
                result = True
                # if result and not token in self.rel_freq:
                #     print(token)

            if self.in_list_freq_words_default(normalize(token)) or self.in_list_freq_words_default(token):
                result = True
                # if result and not token in self.rel_freq:
                #     print(token)

        if result and not token in self.rel_freq:
            if not token  in self.set_in_list:
                self.set_in_list.add(token)
                print(token)

        return result



    def in_list_freq_words_default(self, token):
        return token.lower() in self.list_frequents_words


def collapse_wordpiece(tokens, attentions):
    tokens_aux, attentions_aux, pos = [], [], 0
    for i in range(len(tokens)):
        if tokens[i].startswith("##"):
            tokens_aux[len(tokens_aux)-1] = tokens_aux[len(tokens_aux)-1]+tokens[i][2:]
            attentions_aux[len(attentions_aux)-1].append(attentions[i])
        else:
            tokens_aux.append(tokens[i])
            attentions_aux.append([attentions[i]])
    for i in range(len(attentions_aux)):
        attentions_aux[i] = statistics.mean(attentions_aux[i])
    # st = ""
    # for i in range(len(tokens)):
    #     st+= " " + tokens[i] + str(round(attentions[i],2))
    # print(st)
    # st = ""
    # for i in range(len(tokens_aux)):
    #     st+= " " + tokens_aux[i]+ str(round(attentions_aux[i],2))
    # print(st)
    return tokens_aux, attentions_aux


class Distort():
    def __init__(self):
        self.dict_token_distortion = {}
        self.text = []

    def distort_text(self, text_distortion, string, tokenizer=None):
        ll = []
        cont = 0
        if d_mode["attention"] and text_distortion.analyser:
            mask_beto = self.distort_with_attention(text_distortion,string)
            return mask_beto
        def parche_tokenizer(w):
            if '"' in w:
                tokens = tokenizer(w)
                for i in range(len(tokens)):
                    if tokens[i] == u'``' or tokens[i] == u'\'\'':
                        tokens[i] = u'"'
                return tokens
            else:
                if w=="y":
                    return ["y"]
                return tokenizer(w)

        if isinstance(string,tuple):
            print(string)
        for w in string.split():
            parch = parche_tokenizer(w)
            self.text.extend(parch)
            dist = text_distortion.distort_document(parch)
            ll.append([dist, ' '])
            for i in range(len(dist)):
                token = parch[i]
                mask = dist[i]
                if token != mask:
                    cont = cont + 1
                if token not in self.dict_token_distortion:
                    self.dict_token_distortion[token] = []
                self.dict_token_distortion[token].append(mask)
        import itertools
        string = ""
        list_tokens = list(itertools.chain(*list(itertools.chain(*ll))))
        for token in list_tokens:
            string = "".join([string, token])


        return string

    def distort_with_attention(self, text_distortion, string):
        result = None
        for item in text_distortion.analyser.l_items:
            if item.text == string:
                tokens = []
                aux_att, attentions = item.get_tokens_att(), []
                for i in range(len(item.tokens)):
                    if item.tokens[i] in ["[PAD]" ,  "[SEP]", "[CLS]", "[UNK]"]:
                            continue


                    tokens.append(item.tokens[i])
                    if item.tokens[i] in stopwords.words('spanish') or item.tokens[i] in ",.;:-¿?" or item.tokens[i].isnumeric():
                        attentions.append(0)
                    else:
                        attentions.append(aux_att[i])

                tokens,attentions = collapse_wordpiece(tokens,attentions)

                percentage_to_mask = 10 # less scored tokens

                l_to_sort = [(i, tokens[i], attentions[i]) for i in range(len(tokens) ) if attentions[i]!=0]
                l_sorted = list(sorted(l_to_sort, key=lambda x: x[2]))
                cant_to_mask = (percentage_to_mask*len(l_to_sort))//100

                mean_att = statistics.mean([a for a in attentions if a != 0])
                dest = statistics.stdev([a for a in attentions if a != 0])
                #masking
                result,result_att = "", ""
                i_to_mask = [i for i, token, att in l_sorted[:cant_to_mask]]
                cont_masked = 0
                for i in range(len(tokens)):
                    space = "" if tokens[i].startswith("##") else " "
                    tokens[i] = tokens[i].replace("##","")
                    if  attentions[i] < abs(mean_att-dest)  :
                        result = result + space + ("*"*len(tokens[i]))
                        result_att = result_att + space + ("*"*len(tokens[i])) + "({0})".format(round(attentions[i],4))
                        cont_masked+=1
                    else:
                        result = result + space +tokens[i]
                        result_att = result_att + space + tokens[i] + "({0})".format(round(attentions[i],4))
                if cont_masked < 3:
                    print(string)
                    print(result)
                    print(mean_att-dest,mean_att,dest)



        return result

def freqwordsFCE(a_corp, b_corp, k, tokenizer=None):
    seek_freq_words_s = TextDistortion.SeekFrequentsWords(
        [tokenizer(doc) for doc in a_corp])
    seek_freq_words_t = TextDistortion.SeekFrequentsWords(
        [tokenizer(doc) for doc in b_corp])
    seek_freq_words_s.most_frequents(k=-1),seek_freq_words_t.most_frequents(k=-1)
    # print(seek_freq_words_s.most_frequents(k=100))
    # print()
    # print(seek_freq_words_t.most_frequents(k=100))
    # raise()
    words_freq = TextDistortion.SeekFrequentsWords.FCE(seek_freq_words_s, seek_freq_words_t)


    freq_words = [w for w, _ in words_freq[:k]]
    # print(freq_words)

    return freq_words

def create_distorcion_instance(type_distortion, s_corp=None, validation_corpus=None, k=0,mask_frequents=True, tokenizer=None, optional_list_fw=[],vocab=None):
    freq_words = set(optional_list_fw)

    if d_mode['topic']:
        from nltk.corpus import stopwords
        freq_words = freq_words.difference(stopwords.words('spanish'))
    else:
        if optional_list_fw != []:
            # print('using optional list of fw', optional_list_fw)
            # freq_words = optional_list_fw
            freq_words = freq_words.union(optional_list_fw)#.difference(freq_words)
            # print('using optional list of fw without stopwords', freq_words)
            if (mask_frequents and not to_mask_sw) or (to_mask_sw and not mask_frequents):
                from nltk.corpus import stopwords
                print("stopwords EXCLUDED as frequent words")
                freq_words = freq_words.difference(stopwords.words('spanish'))
        else:
            from nltk.corpus import stopwords
            # stopwords, not always I want to concider it as frequents, depends on whether I want to mask or not
            if (mask_frequents and to_mask_sw) or ((not mask_frequents) and (not to_mask_sw)):
                # nltk.download('stopwords')
                print("stopwords INCLUDED as frequent words")
                freq_words = freq_words.union(stopwords.words('spanish'))
            else:
                print("stopwords EXCLUDED as frequent words")
                freq_words = freq_words.difference(stopwords.words('spanish'))

    print("FINALLY there are {0} words to mask/keep".format(len(freq_words)))
    return TextDistortion(list_frequents_words=freq_words, fun=type_distortion,mask_frequents=mask_frequents,from_BETO=d_mode['attention'],vocab=vocab)
