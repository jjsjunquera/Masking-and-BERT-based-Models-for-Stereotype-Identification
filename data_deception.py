import os





def read_(name, for_split):
    path = "C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\\"
    path = ""
    l_text,l_label, l_code =[],[],[]
    for aux,tag in [('.False',0),('.True',1)]:
        with open(path+name+aux,encoding="utf8") as f:
            lines = f.read().split(for_split)
            for line in lines:
                code = line.split('\t ')[0]
                text = line[len(line.split('\t ')[0])+len('\t '):]
                if text == '':
                    continue
                l_text.append(text)
                l_label.append(tag)
                l_code.append(code)
        print(len(l_text))
        if len(l_code) != len(set(l_code)):
            raise Exception("not ids")
    return l_text,l_label,l_code

def read_rev_hotel():
    return read_("hotel","\n&&&\n")
def read_rev_doctor():
    return read_("doctor","\n&&&\n")
def read_rev_restaurant():
    return read_("restaurant","\n&&&\n")

def read_contr_ab():
    return read_("abortion","abortion.")
def read_contr_bf():
    return read_("bestFriend","bestFriend.")
def read_contr_dp():
    return read_("deathPenalty","deathPenalty.")









def read(dir,prefix_name, l_text):
        contenido = os.listdir(dir)
        for fichero in contenido:
            if os.path.isfile(os.path.join(dir, fichero)) and fichero.endswith('.txt'):
                with open(os.path.join(dir, fichero),encoding="utf8") as f:
                    print(os.path.join(dir, fichero))
                    l_text.append(str(len(l_text)+1)+prefix_name+fichero+'\t '+f.read())

def create_hotel_one_file():
    l_text = []
    read(dir='C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\deception\deception_dataset\hotel\\negative\deceptive_turker', prefix_name=".negative_deceptive_turker_",l_text=l_text)
    read(dir='C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\deception\deception_dataset\hotel\\positive\deceptive_turker', prefix_name=".positive_deceptive_turker_",l_text=l_text)
    f = open("hotel.False", "w")
    for text in l_text[:-1]:
        f.write(text+"\n&&&\n")
    f.write( l_text[-1])
    f.close()

    l_text = []
    read(dir='C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\deception\deception_dataset\hotel\\negative\\truthful', prefix_name=".negative_truthfulr_",l_text=l_text)
    read(dir='C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\deception\deception_dataset\hotel\\positive\\truthful', prefix_name=".positive_truthful_",l_text=l_text)
    f = open("hotel.True", "w")
    for text in l_text[:-1]:
        f.write(text+"\n&&&\n")
    f.write( l_text[-1])
    f.close()

    l_text = []
    read(dir='C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\deception\deception_dataset\doctor\\deceptive_MTurk', prefix_name=".doctor_deceptive_MTurk_",l_text=l_text)
    f = open("doctor.False", "w")
    for text in l_text[:-1]:
        f.write(text+"\n&&&\n")
    f.write( l_text[-1])
    f.close()
    l_text = []
    read(dir='C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\deception\deception_dataset\doctor\\truthful', prefix_name=".doctor_truthful_",l_text=l_text)
    f = open("doctor.True", "w")
    for text in l_text[:-1]:
        f.write(text+"\n&&&\n")
    f.write( l_text[-1])
    f.close()

    l_text = []
    read(dir='C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\deception\deception_dataset\\restaurant\\deceptive_MTurk', prefix_name=".restaurant_deceptive_MTurk_",l_text=l_text)
    f = open("restaurant.False", "w")
    for text in l_text[:-1]:
        f.write(text+"\n&&&\n")
    f.write( l_text[-1])
    f.close()
    l_text = []
    read(dir='C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\deception\deception_dataset\\restaurant\\truthful', prefix_name=".restaurant_truthful_",l_text=l_text)
    f = open("restaurant.True", "w")
    for text in l_text[:-1]:
        f.write(text+"\n&&&\n")
    f.write( l_text[-1])
    f.close()





d_name_f = {"read_contr_ab":read_contr_ab,"read_contr_bf":read_contr_bf,"read_contr_dp":read_contr_dp,
            "read_rev_hotel":read_rev_hotel,"read_rev_doctor":read_rev_doctor,"read_rev_restaurant":read_rev_restaurant}
