from StereotypeCorpus import StereotypeCorpus


def get_corpus(task, statistics=False, use_lr=False):
    print("---------LOADING CORPUS-------------")
    corpus = StereotypeCorpus(task)
    # corpus.load_instances_from_excel(["CAT_TOT_solo5_N_R_BUENO.xlsx"])    
    # corpus.load_instances_label88(["LABEL_88.xlsx"])
    corpus.load_hyp_from_excel(["demo.xlsx"])
    
    if statistics or use_lr:
        print("\n\nSTATISTICS:")
        corpus.statistics( use_lr=use_lr)
        print("\n\n----------------------:")
    return corpus



