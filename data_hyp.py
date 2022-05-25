# Import `os`
import matplotlib
import os

# Retrieve current working directory (`cwd`)
from emoji import unicode_codes, emoji_count
from nltk import FreqDist, word_tokenize
from xml.dom import minidom

from utils import clusters


def data():
    # cwd = os.getcwd()
    # cwd
    # path = "C:\\Users\\Juan Javier\\Google Drive (junquera1988@gmail.com)\\doctorado\\Bias in IR\\hyperpartisan\\data\\a stylometri inquiry into hyperpartisan and fake news\\articles"
    path = "articles"
    # Change directory
    os.chdir(path)

    # List all files and directories in current directory
    # print(os.listdir('.'))
    from os import walk

    path_name_list = [(path, name) for path, _, name in walk("articles")]
    l_data = []
    count_double_texts,count_here,count_empty_text = 0,0,0
    contradiction = []
    for path, d in path_name_list:
        for name in d:
            doc_text = minidom.parse(path + "\\" + name)
            if doc_text.getElementsByTagName("mainText")[0].firstChild is None:
                # print(doc.name, "has not mainText")
                count_empty_text +=1
                continue
            else:
                main_text = doc_text.getElementsByTagName("mainText")[0].firstChild.data

            if doc_text.getElementsByTagName("veracity")[0].firstChild is None:
                # print(doc.name, "has not veracity tag")

                # ---- comment one of them
                # return
                veracity = None
                # ----
            else:
                veracity = doc_text.getElementsByTagName("veracity")[0].firstChild.data

            if doc_text.getElementsByTagName("orientation")[0].firstChild is None:
                # print(doc.name, "has not orientation tag")

                # ---- comment one of them
                continue
                # orientation = None
                # ----

            else:
                orientation = doc_text.getElementsByTagName("orientation")[0].firstChild.data
                # if orientation!="mainstream": for hyp vs main
                #     orientation="hyperpartisan"
                # if orientation=="mainstream": # for left vs right
                #     continue

            if doc_text.getElementsByTagName("portal")[0].firstChild is None:
                print(name)
                raise Exception("without portal")
            else:
                portal = doc_text.getElementsByTagName("portal")[0].firstChild.data

            if 'The document has moved here.' == main_text:
                count_here+=1
                continue

            # checking double texts

            save_this_document = True
            for d in l_data:
                if d["text"]==main_text:
                    # print(count_double_texts, main_text)
                    count_double_texts+=1
                    if d["veracity"] != veracity or d["orientation"] != orientation:
                        # print("----- These documents has the same text and different labels: they will be removed -----")
                        # print(d["name"], name)
                        # print("-----")
                        contradiction.append(d)
                    save_this_document = False
                    break

            if not save_this_document:
                continue

            l_data.append({"name": name,
                           "text": main_text,
                           "orientation": orientation,
                           "veracity": veracity,
                           "portal": portal})
    # print(count_double_texts, "double texts")
    # print(count_here, "here")
    # print(count_empty_text, "here")
    for d in contradiction:
        if d in l_data:
            l_data.remove(d)
    print(len(l_data),"documents")
    # from balancing_data import clusters
    clustered_data = clusters(l_data, f_att=lambda x: x["orientation"])

    return l_data

data()
