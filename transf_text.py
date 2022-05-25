import nltk
from sklearn.feature_extraction.text import CountVectorizer, _document_frequency

from setting import ngram_range, analyzer, k_most_fw, mask_frequents
from text_distortion import create_distorcion_instance, TextDistortion, Distort


def transforming_texts(training_set, test_set, k=k_most_fw, ngram_range=ngram_range, optional_list_fw=[]):
    # from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = CountVectorizer(analyzer=analyzer, binary=False, ngram_range=ngram_range, min_df=3)
    text_distortion = create_distorcion_instance(TextDistortion.dv_ma, mask_frequents=mask_frequents, optional_list_fw=optional_list_fw)
    distort = Distort()
    tokenizer=vectorizer.build_tokenizer()


    aux = [distort.distort_text(text_distortion=text_distortion, string=elem, tokenizer=tokenizer) for elem in training_set]



    transformed_train = vectorizer.fit_transform(aux)

    y_train = [elem[1] for elem in training_set]
    y_test = [elem[1] for elem in test_set]



    transformed_test = vectorizer.transform(
        [distort.distort_text(text_distortion=text_distortion, string=elem[0], tokenizer=tokenizer) for elem in
         test_set])

    #  for non masking at all
    # transformed_train = vectorizer.fit_transform([elem[0] for elem in training_set])
    # transformed_test = vectorizer.transform([elem[0] for elem in test_set])
    return transformed_train, y_train, transformed_test, y_test, vectorizer

def apply_mask(l_text,l_mostFrequent, vocab=None):
    
    vectorizer = CountVectorizer(analyzer=analyzer, binary=False, ngram_range=ngram_range, min_df=3)
    distortion_type = TextDistortion.dv_ma
    # distortion_type = TextDistortion.dv_sa
    # distortion_type = TextDistortion.dv_l2n
    # distortion_type = TextDistortion.dv_ex
    text_distortion = create_distorcion_instance(distortion_type, mask_frequents=mask_frequents, optional_list_fw=l_mostFrequent,vocab=vocab)
    distort = Distort()
    tokenizer=vectorizer.build_tokenizer()
    result = [distort.distort_text(text_distortion=text_distortion, string=elem.lower(), tokenizer=tokenizer) for elem in l_text]
    # print("Examples of masking")
    # for i in range(len(l_text[:5])):
    #     print(l_text[i])
    #     print(result[i])
    #     print()


    return result
    
