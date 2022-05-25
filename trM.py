## bow.py
import numpy
# from pandas.tests.sparse.frame.test_to_from_scipy import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from stop_words import get_stop_words


from setting import ngram_range, analyzer, k_most_fw, min_df, d_mode
from text_distortion import freqwordsFCE
from transf_text import apply_mask
from utils import tokenizer_for_alpha


def bow_count_occurrences():
    count_vec = CountVectorizer(stop_words=get_stop_words('spanish'),min_df=3,binary=False, tokenizer=tokenizer_for_alpha)
    # count_vec = CountVectorizer(analyzer=analyzer, binary=False, ngram_range=ngram_range, min_df=3)
    return count_vec

def bow_count_occurrencesMASK():
    count_vec = CountVectorizer(analyzer=analyzer, binary=False, ngram_range=ngram_range, min_df=min_df)
    print("CounterVectorizer at Masking is with",ngram_range, analyzer)
    return count_vec




def prep(x):
    return x

def build_model(mode, cls):
    # Intent to use default paramaters for show case
    vect = None
    if mode == 'count':
        vect = bow_count_occurrences()
    elif mode=='masked':
        vect = bow_count_occurrencesMASK()

    else:
        raise ValueError('Mode should be either count or tfidf')
    params = [('vect', vect)]
    if cls == "lr":
        params.append(('cls',LogisticRegression(solver='newton-cg', n_jobs=-1)))
    elif cls == "svm":
        from sklearn.svm import LinearSVC
        params.append(('cls', LinearSVC()))
    elif cls == "nb":
        params.append(('tr', FunctionTransformer(lambda x: x.todense(), accept_sparse=True)))
        params.append(('cls', GaussianNB()))
    elif cls == "rf":
        params.append(('cls', RandomForestClassifier()))
    else:
        raise ValueError('Cls should be set')
    return Pipeline(params)


def preprocess_x(df):
    return df

def preprocess_y(y):
    return y

def print_conf_matrix(y, y_pred):
    import pandas as pd
    data = {'y_Actual': y,
            'y_Predicted': y_pred
            }
    df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
    confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

    # print(confusion_matrix)
    print(classification_report(y, y_pred))
    # import seaborn as sn
    # sn.heatmap(confusion_matrix, annot=True)
    # import matplotlib.pyplot as plt
    # plt.show()


def mutual_info(x, y, classif, k):
    X_vec = classif.fit_transform(x)
    res = {}
    print("\n.....Mutual Information.....")
    result = []
    for label in set(y):

        res[label] = dict(zip(classif.get_feature_names(),
                              mutual_info_classif(X_vec, [label if label == l else "0" for l in y], copy=True)
                              ))
        print()
        print(label)
        cont = 0
        result.extend([w for w,_ in sorted(res[label].items(), key=lambda w_s: w_s[1], reverse=True)][:k//2])
        for w, score in sorted(res[label].items(), key=lambda w_s: w_s[1], reverse=True):
            if cont == k:
                break
            print(w)
            cont += 1
    return set(result)




def conf_examples(x, y, y_pred, l_code, corpus):
    corpus.d_code_maskresults = {}
    for i in range(len(x)):
        corpus.d_code_maskresults[l_code[i]] = {"masked_text": x[i],
                                              "actual": y[i],
                                              "predicted": y_pred[i]}
    return



    l_act_pred_text = []
    for i in range(len(x)):
        if y[i]!=y_pred[i]:
            l_act_pred_text.append((y[i],y_pred[i],x[i]))
    # ff = open("/home/jjsjunquera/Stereotype/output/final_predictionF", "w")
    ind=1
    for act, pred, ex in list(sorted(l_act_pred_text,key=lambda t:len(t[2]),reverse=False))[10:20]:
       # ff.write("\n\n"+'{} & {}  & {}'.format(act, pred, ex))
        print(str(ind),ex)
        ind+=1
        print()
    # ff.close()

    print("\n\n")
    l_act_pred_text = []
    for i in range(len(x)):
        if y[i]==y_pred[i] and y[i]==0:
            l_act_pred_text.append((y[i],y_pred[i],x[i]))
    # ff = open("/home/jjsjunquera/Stereotype/output/final_predictionF", "w")
    ind=1
    for act, pred, ex in list(sorted(l_act_pred_text,key=lambda t:len(t[2])))[10:20]:
       # ff.write("\n\n"+'{} & {}  & {}'.format(act, pred, ex))
       #  print("\n"+'{} & {}  & {} & {}  & {}'.format(act, corpus.get_tagOrig_by_code(d_text_code[ex]) ,pred,d_text_code[ex]))
       print(str(ind),ex)
       ind+=1
       print()
    # ff.close()

    print("\n-----\n")
    l_act_pred_text = []
    ind=1
    for i in range(len(x)):
        if y[i]==y_pred[i] and y[i]==1:
            l_act_pred_text.append((y[i],y_pred[i],x[i]))
    # ff = open("/home/jjsjunquera/Stereotype/output/final_predictionF", "w")
    for act, pred, ex in list(sorted(l_act_pred_text,key=lambda t:len(t[2])))[10:20]:
       # ff.write("\n\n"+'{} & {}  & {}'.format(act, pred, ex))
       #  print("\n"+'{} & {}  & {} & {}  & {}'.format(act, corpus.get_tagOrig_by_code(d_text_code[ex]) ,pred,d_text_code[ex]))
       print(str(ind),ex)
       ind+=1
       print()
    # ff.close()


def pipeline(l_code, x, y, mode, cls, corpus):
    x = preprocess_x(x)
    y = preprocess_y(y)

    model_pipeline = build_model(mode, cls)

    cv = KFold(n_splits=10, shuffle=True)
    y_pred = cross_val_predict(model_pipeline, x, y, cv=10)

    # mutual_info(x, y, build_model(mode,cls)[0])
    conf_examples(x,y,y_pred, l_code, corpus)
    print_conf_matrix(y, y_pred)



    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

    X_train = model_pipeline[0].fit_transform(x)
    names = model_pipeline[0].get_feature_names()
    lr = LogisticRegression(C=1.0).fit(X_train, y)
    coefs=lr.coef_[0]

    top_three_sorted=numpy.argsort(coefs)[-50:]
    print("Total number of features",len(model_pipeline[0].vocabulary_))

    # print([names[i] for i in top_three_sorted])
    return model_pipeline


def vocabulary_to_handle(x_train, y_train):
    x = preprocess_x(x_train)
    y = y_train
    model_pipeline = build_model(mode='count',cls='lr')
    model_pipeline.fit(x, y)
    print('Number of Vocabulary: %d' % (len(model_pipeline.named_steps['vect'].get_feature_names())))

def get_mostfrequent(corpus):
    textslabel1, textslabel2 = corpus.get_d_label_texts().values()
    mf = freqwordsFCE(textslabel1, textslabel2, k_most_fw, tokenizer=corpus.tokenize_text)
    return mf
    
def classify(X, y, l_code, corpus, attention=None, analyser=None):
    # X = [text.lower() for text in X]
    # using_deepl(x_test, y_test, l_code)
    l_cls = "lr svm nb rf".split(" ")
    l_mode = "masked count tfidf".split(" ")
    for mode in l_mode[:1]:
        for cls in l_cls[:1]:
            print()
            print('Using', mode, "with", cls, "-------")
            if mode=='masked':
                mf = []
                vocab = None
                rel_freq = []
                if not d_mode['diference_per_label'] and len(set(y))==2:
                    d_mode['diference_per_label']=True
                    rel_freq = get_mostfrequent(corpus)
                    d_mode['diference_per_label']=False


                if k_most_fw != None:
                    print("MASKING BEYOND SW")
                    if d_mode['topic'] or d_mode['fw_in_corpus']:
                        mf = corpus.most_freq(k_most_fw)
                    if d_mode['fce'] or d_mode['diference_per_label']:
                        mf = get_mostfrequent(corpus)
                    if d_mode['mi']:
                        mf = mutual_info(X, y, CountVectorizer(min_df=3,binary=False), k=k_most_fw//2)
                        # mf = lr_weighted_words[:]
                    if d_mode['attention']:
                        c1, c2, vocab = attention
                        # mf = c1[:500]+c2[:500]
                        mf = c1[:5000]



                    print(len(mf))

                pipeline(l_code, apply_mask(X, l_mostFrequent = mf, vocab=(vocab,rel_freq,analyser)), y, mode=mode, cls = cls, corpus=corpus)


               
            else:
                pipeline(l_code, X, y, mode=mode, cls = cls,corpus=corpus)
