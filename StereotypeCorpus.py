import random
from datetime import datetime

import xlrd
import xlsxwriter

from Row import Row
from corpus import CreateCorpus
from dataset.SERVIPOLI.imm_keywords import speech_contains_key
from trM import vocabulary_to_handle, classify
from utils import clusters
import itertools
import nltk
import spacy
# import es_core_news_sm


def load_spacy():
    print("\nSPANISH SPACY\n")
    return spacy.load("es_core_news_sm")
    # print("\nENGLISH SPACY\n")
    # return spacy.load("en_core_web_sm")


class StereotypeCorpus:
    def __init__(self, task):
        self.d_code_index = {}
        self._l_rows = None
        self._d_tax1_instances = None
        self.task = task
        self.nlp = None

    def get_rows(self):
        return self._l_rows[:]

    def load_hyp_from_excel(self, excel_files):
        self._l_rows = CreateCorpus.load_rows_from_hyp(excel_files)
        cont_duplicated = 0
        for instance_index in range(len(self._l_rows)):
            if self._l_rows[instance_index].cod in self.d_code_index:
                # print(self._l_rows[instance_index].cod, instance_index, self.d_code_index[self._l_rows[instance_index].cod])
                cont_duplicated += 1
            else:
                self.d_code_index[self._l_rows[instance_index].cod] = instance_index
        if cont_duplicated > 0:
            print(cont_duplicated, "duplicated codes")
        print(len(self.d_code_index), "tagged examples where loaded")

    def load_instances_from_excel(self, excel_files):
        self._l_rows = CreateCorpus.load_rows_from_spss(excel_files)

        cont_duplicated = 0
        for instance_index in range(len(self._l_rows)):
            if self._l_rows[instance_index].cod in self.d_code_index:
                # print(self._l_rows[instance_index].cod, instance_index, self.d_code_index[self._l_rows[instance_index].cod])
                cont_duplicated += 1
            else:
                self.d_code_index[self._l_rows[instance_index].cod] = instance_index
        if cont_duplicated > 0:
            print(cont_duplicated, "duplicated codes")
        print(len(self.d_code_index), "tagged examples where loaded")

    def get_original_tag(code):
        return

    def load_instances_label88(self, excel_files):
        aux = CreateCorpus.load_rows(excel_files, label88=True)
        l_to_ignore = ["5201-9", "1105-28", "1682-4", "4743-7", "4752-4", "1105-32", "4711-34", "4743-23", "1105-33",
                       "4743-20",
                       "1859-14", "4713-48", "4743-21", "911-1", "4457-6", "834-19", "1105-14", "1105-24", "4711-33",
                       "707-9",
                       "4711-36", "4711-79", "4711-88", "1105-29", "4317-3", "778-1", "881-5", "2075-1", "2558-35",
                       "4365-1",
                       "4393-23", "4987-8", "899-1", "1003-18", "1337-21", "4393-11", "4393-40", "4453-56", "4457-1",
                       "1105-11",
                       "834-1", "552-1", "1118-1", "829-3", "846-3", "1434-15", "1859-1"]
        print(len(l_to_ignore), "To ignore from 88 cause size and the expert valoration")
        if self._l_rows == None:
            self._l_rows = []
        i = len(self._l_rows)
        self._l_rows.extend(aux)
        dup = []
        cont_duplicated = 0
        for instance_index in range(i, len(self._l_rows)):
            if self._l_rows[instance_index].cod in l_to_ignore:
                continue
            if self._l_rows[instance_index].cod in self.d_code_index:
                # print(self._l_rows[instance_index].cod, instance_index, self.d_code_index[self._l_rows[instance_index].cod])
                cont_duplicated += 1
                dup.append(self._l_rows[instance_index].cod)
            else:
                self.d_code_index[self._l_rows[instance_index].cod] = instance_index
        if cont_duplicated > 0:
            print(cont_duplicated, "duplicated codes with 88", dup)
        print(len(self.d_code_index), "tagged examples where loaded")

    def get_tax1(self):
        if self._d_tax1_instances is None:
            self._d_tax1_instances = clusters(self.d_code_index, f_att=lambda cod: self.task.to_label(
                self._l_rows[self.d_code_index[cod]].tax1, cod))
            d_c_text = {}
            for label in self._d_tax1_instances:
                d_c_text[label] = [(code, self.get_text_by_code(code)) for code in self._d_tax1_instances[label]]
            print(set(self._d_tax1_instances))
            self.adjusting_classes(self._d_tax1_instances)
        return self._d_tax1_instances

    def statistics(self, use_lr=False):
        # number of labeled instances
        print("tax1:",
              sum([len(l_row_index) for label, l_row_index in self.get_tax1().items() if label is not None]))
        # distribution of labels
        for label, l_codes in self.get_tax1().items():
            if label is not None:
                print(label, "with", len(l_codes), "examples")

        # print("\n---vocabulary---")
        # self.vocabulary()
        # print()

        x_test, y_test, l_code = self.get_texts_and_labels(f_get_tax=self.get_tax1)
        if use_lr:
            classify(x_test, y_test, l_code, corpus=self)
        # print()

    def vocabulary(self):
        import spacy
        import es_core_news_sm
        if self.nlp == None:
            self.nlp = load_spacy()
        print("vocabulary tax 1")
        l_text, l_label, l_code = self.get_texts_and_labels(f_get_tax=self.get_tax1)
        # vocabulary_to_handle(x_train=l_text, y_train=l_label)
        sizes_text = [(len([token for token in self.nlp(text) if token.pos_ != "PUNCT"]), text) for text in l_text]
        sizes = [s for s, _ in sizes_text]
        print("min size", min(sizes))
        print("max size", max(sizes))
        import statistics
        average = statistics.mean(sizes)
        dest = statistics.stdev(sizes)
        print("mean", average, "stdev", dest)
        print('number of texts greater than average', len([1 for s in sizes if s > average]))
        print('number of texts less than average', len([1 for s in sizes if s <= average]))
        print('number of texts greater than average+stdev', len([1 for s in sizes if s > average + dest]))

        # words = ['hello', 'hell', 'owl', 'hello', 'world', 'war', 'hello', 'war']
        # freq = nltk.FreqDist(words)
        # print(freq.most_common(3))

    def most_freq(self, n):
        if self.nlp == None:
            self.nlp = load_spacy()
        l_text, l_label, l_code = self.get_texts_and_labels(f_get_tax=self.get_tax1)
        tokens = [[token.text for token in self.nlp(text1) if token.pos_ != "PUNCT"] for text1 in l_text]
        all_tokens = list(sorted(itertools.chain.from_iterable(tokens), key=lambda x: x))
        freq = nltk.FreqDist(all_tokens)
        return [w for w, _ in freq.most_common(n)]

    def tokenize_text(self, text):
        if self.nlp == None:
            self.nlp = load_spacy()
        return [token.text for token in self.nlp(text) if token.pos_ != "PUNCT"]

    def get_d_label_texts(self):
        l_text, l_label, l_code = self.get_texts_and_labels(f_get_tax=self.get_tax1)
        d_label_texts = {}
        for i in range(len(l_text)):
            if l_label[i] not in d_label_texts:
                d_label_texts[l_label[i]] = []
            d_label_texts[l_label[i]].append(l_text[i])
        return d_label_texts

    def get_texts_and_labels(self, f_get_tax):
        l_text = []
        l_label = []
        l_codes = []
        for label, l_code in f_get_tax().items():
            for code in l_code:
                l_label.append(label)
                l_text.append(self.get_text_by_code(code))
                l_codes.append(code)
        return l_text, l_label, l_codes

    def get_text_by_code(self, code):
        index = self.d_code_index[code]
        return self._l_rows[index].sentence

    def get_tagOrig_by_code(self, code):
        index = self.d_code_index[code]
        return self._l_rows[index].tax1_orig

    def adjusting_classes(self, d_tax1_instances):
        # label_negative = 1
        print("before", d_tax1_instances.keys())
        for no_label in {None}:
            if no_label in d_tax1_instances:
                d_tax1_instances.pop(no_label)
        # for label in list(d_tax1_instances.keys()):
        #     if label not in {"2.2","2.1"}:
        #         d_tax1_instances.pop(label)
        minimum = min([len(values) for values in d_tax1_instances.values()])
        for k in d_tax1_instances:
            aux = d_tax1_instances[k]
            random.shuffle(aux)
            d_tax1_instances[k] = aux[:]
        print("after", d_tax1_instances.keys())

    def create_file_corpus_speeches(self):
        import xlsxwriter
        # Create a workbook and add a worksheet.
        workbook = xlsxwriter.Workbook('StereoImmigrants_at_speech_level.xlsx')
        worksheet = workbook.add_worksheet()
        # Add a bold format to use to highlight cells.
        bold = workbook.add_format({'bold': 1})
        worksheet.write(0, 0, 'Code', bold)
        worksheet.write(0, 1, 'Speaker', bold)
        worksheet.write(0, 2, 'Party', bold)
        worksheet.write(0, 3, 'Date', bold)
        worksheet.write(0, 4, '1', bold)
        worksheet.write(0, 5, '2', bold)
        worksheet.write(0, 6, '3', bold)
        worksheet.write(0, 7, '4', bold)
        worksheet.write(0, 8, '5', bold)
        worksheet.write(0, 9, '88', bold)
        worksheet.write(0, 10, 'Speech', bold)
        speeches = clusters(self._l_rows, f_att=lambda r: r.cod1)
        r = 1
        for cod1, l_row in speeches.items():
            speaker, speech, party, date = l_row[0].speaker, l_row[0].speech, l_row[0].party, l_row[0].date
            if "president" in speaker.lower():
                continue
            labels = [row.tax1 for row in l_row]
            worksheet.write(r, 0, cod1)
            worksheet.write(r, 1, speaker)
            worksheet.write(r, 2, party)
            worksheet.write(r, 3, date)
            worksheet.write(r, 4, labels.count('1'))
            worksheet.write(r, 5, labels.count('2'))
            worksheet.write(r, 6, labels.count('3'))
            worksheet.write(r, 7, labels.count('4'))
            worksheet.write(r, 8, labels.count('5'))
            worksheet.write(r, 9, labels.count('88'))
            worksheet.write(r, 10, speech)
            r += 1
        workbook.close()


def load_servipoli(source_files, path, d_result_servipoli):
    for fil_name in source_files:
        wb = xlrd.open_workbook(path + '\\' + fil_name + ".xlsx")
        sheet = wb.sheet_by_index(0)
        print("Reading from sheet:", sheet.name)
        print("A total of", len(list(sheet.get_rows())), "rows")
        for row in list(sheet.get_rows())[1:]:
            row = Row(excel_row=row, read_servipoli=True)
            if row.cod not in d_result_servipoli:
                d_result_servipoli[row.cod] = []
            d_result_servipoli[row.cod].append((row, fil_name))
    for cod, l in d_result_servipoli.items():
        if len(l) != 5:
            print(cod, "ignorado por estar repetido el código")


def load_prueba_poli(path, d_result_servipoli={}):
    l_row = []
    d = {}
    wb = xlrd.open_workbook(path + '\\' + "Prueba_poli_COMPLETA" + ".xlsx")
    sheet = wb.sheet_by_index(0)
    print("Reading from sheet:", sheet.name)
    print("A total of", len(list(sheet.get_rows())), "rows")
    for row in list(sheet.get_rows())[1:]:
        row = Row(excel_row=row, read_prueba_poli_COMPLETA=True)
        if row.cod in d_result_servipoli:
            raise Exception("cod rep en servipoli y prueba_poli")
        if row.cod not in d:
            d[row.cod] = row
        else:
            print("duplicados en prueba poli", row.cod)
            continue
        l_row.append(row)
    return l_row


def load_Corp_congreso_v2(param):
    l_row = []
    wb = xlrd.open_workbook(param)
    sheet = wb.sheet_by_index(0)
    print("Reading from sheet:", sheet.name)
    print("A total of", len(list(sheet.get_rows())), "rows")
    for row in list(sheet.get_rows())[1:]:
        try:
            row = Row(excel_row=row, read_Corp_congreso_v2=True)
            l_row.append(row)
        except:
            pass
    return l_row

def load_immspeech(param):
    l_row = []
    wb = xlrd.open_workbook(param)
    sheet = wb.sheet_by_index(0)
    print("Reading from sheet:", sheet.name)
    print("A total of", len(list(sheet.get_rows())), "rows")
    for row in list(sheet.get_rows())[1:]:
        row = Row(excel_row=row, read_immspeech=True)
        l_row.append(row)
    return l_row


def add_pruebapoli_tosheet_agreement(l_row, worksheet, r, d_codespeech, from_file=None):
    for l in l_row:
        worksheet.write(r, 0, l.cod1)
        worksheet.write(r, 1, l.cod2)
        worksheet.write(r, 2, l.sentence)
        worksheet.write(r, 3, l.tax1)
        col = 4
        tags_first_digit, eva_first_digit = [], []
        for i in range(len(l.cat)):
            worksheet.write(r, col, str(l.cat[i]))
            col += 1
            if len(str(l.cat[i])) >= 1:
                tags_first_digit.append(str(l.cat[i])[0])
        for i in range(len(l.eval)):
            worksheet.write(r, col, str(l.eval[i]))
            col += 1
            if len(str(l.eval[i])) >= 1:
                eva_first_digit.append(str(l.eval[i])[0])
        col += 1
        # adding agreement de cat
        s_tags_first_digit = set(tags_first_digit)
        m, a = -1, -1
        for tag in s_tags_first_digit:
            c = tags_first_digit.count(tag)
            if c >= 3:
                m, a = tag, c
        worksheet.write(r, col, m)
        col += 1
        worksheet.write(r, col, a)
        col += 1
        # adding agreement de eva
        s_eva_first_digit = set(eva_first_digit)
        m, a = -1, -1
        for eva in s_eva_first_digit:
            c = eva_first_digit.count(eva)
            if c >= 3:
                m, a = eva, c
        worksheet.write(r, col, m)
        col += 1
        worksheet.write(r, col, a)
        col += 1
        worksheet.write(r, col, from_file)
        col += 1
        worksheet.write(r, col, str(d_codespeech[l.cod1]))
        col += 1
        r += 1
    return r


def ind_corp(l_Corp_congreso_v2, text):
    speech = [r.Column1 for r in l_Corp_congreso_v2 if text.lower() in r.text.lower()]
    if len(speech) == 1:
        return speech[0]
    elif len(speech) >= 1:
        return speech
    else:
        return ""


def merge_servipoli():
    l_Corp_congreso_v2 = load_Corp_congreso_v2(
        "C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\Corp_Congreso_V2.xlsx")
    d_codespeech = {}
    d_result_servipoli = {}
    path = "C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\dataset\SERVIPOLI"
    source_files = ["SERVIPOLI_1", "SERVIPOLI_4", "SERVIPOLI_5", "SERVIPOLI_7", "SERVIPOLI_9"]
    load_servipoli(source_files, path, d_result_servipoli)
    rs = [row[0][0] for row in d_result_servipoli.values()]
    for r in rs:
        text = r.sentence
        if r.cod1 not in d_codespeech or d_codespeech[r.cod1] == "":
            d_codespeech[r.cod1] = ind_corp(l_Corp_congreso_v2, text)
    l_row = load_prueba_poli(path, d_result_servipoli)
    for r in l_row:
        text = r.sentence
        if r.cod1 not in d_codespeech or d_codespeech[r.cod1] == "":
            d_codespeech[r.cod1] = ind_corp(l_Corp_congreso_v2, text)
    l_stereoimmigrant = CreateCorpus.load_rows_from_spss(
        ["C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\CAT_TOT_solo5_N_R_BUENO.xlsx"])
    for r in l_stereoimmigrant:
        text = r.sentence
        if r.cod1 not in d_codespeech or d_codespeech[r.cod1] == "":
            d_codespeech[r.cod1] = ind_corp(l_Corp_congreso_v2, text)

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(path + '\servipoli-1_4_5_7_9.xlsx')
    worksheet = workbook.add_worksheet()

    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': 1})
    worksheet.write(0, 0, 'COD_1', bold)
    worksheet.write(0, 1, 'COD_2', bold)
    worksheet.write(0, 2, 'Sentence', bold)
    worksheet.write(0, 3, 'Tax_1', bold)
    col = 4
    for fil_name in source_files:
        worksheet.write(0, col, 'CAT_' + fil_name, bold)
        col += 1
    for fil_name in source_files:
        worksheet.write(0, col, 'EVA_' + fil_name, bold)
        col += 1
    col += 1
    worksheet.write(0, col, 'CAT_' + fil_name, bold)
    col += 1
    worksheet.write(0, col, 'ACAT_' + fil_name, bold)
    col += 1
    worksheet.write(0, col, 'EVA_' + fil_name, bold)
    col += 1
    worksheet.write(0, col, 'AEVA_' + fil_name, bold)
    col += 1

    r = 1
    r = add_pruebapoli_tosheet_agreement(l_row, worksheet, r, d_codespeech, from_file="Prueba_Poli")
    r = add_pruebapoli_tosheet_agreement(l_stereoimmigrant, worksheet, r, d_codespeech, from_file="StereoImmigrant")
    for cod, l in d_result_servipoli.items():
        if len(l) == 5:
            worksheet.write(r, 0, l[0][0].cod1)
            worksheet.write(r, 1, l[0][0].cod2)
            worksheet.write(r, 2, l[0][0].sentence)
            worksheet.write(r, 3, l[0][0].tax1)
            col = 4
            tags_first_digit, eva_first_digit = [], []
            for i in range(len(l)):
                if l[i][1] != source_files[i]:
                    raise Exception
                worksheet.write(r, col, str(l[i][0].cat))
                col += 1
                if len(str(l[i][0].cat)) >= 1:
                    tags_first_digit.append(str(l[i][0].cat)[0])
            for i in range(len(l)):
                if l[i][1] != source_files[i]:
                    raise Exception
                worksheet.write(r, col, str(l[i][0].eval))
                col += 1
                eva_first_digit.append(str(l[i][0].eval))
            col += 1
            # adding agreement
            s_tags_first_digit = set(tags_first_digit)
            m, a = -1, -1
            for tag in s_tags_first_digit:
                c = tags_first_digit.count(tag)
                if c >= 3:
                    m, a = tag, c
            worksheet.write(r, col, m)
            col += 1
            worksheet.write(r, col, a)
            col += 1
            # adding agreement de eva
            s_eva_first_digit = set(eva_first_digit)
            m, a = -1, -1
            for eva in s_eva_first_digit:
                c = eva_first_digit.count(eva)
                if c >= 3:
                    m, a = eva, c
            worksheet.write(r, col, m)
            col += 1
            worksheet.write(r, col, a)
            col += 1
            worksheet.write(r, col, "servipoli")
            col += 1
            worksheet.write(r, col, str(d_codespeech[l[0][0].cod1]))
            col += 1
            r += 1

    workbook.close()


def load_todos(param):
    l_row = []
    wb = xlrd.open_workbook(param)
    sheet = wb.sheet_by_index(0)
    print("Reading from sheet:", sheet.name)
    print("A total of", len(list(sheet.get_rows())), "rows")
    for row in list(sheet.get_rows())[1:]:
        row = Row(excel_row=row, read_todos=True)
        l_row.append(row)
    return l_row


def load_before2006(param):
    l_row = []
    wb = xlrd.open_workbook(param)
    sheet = wb.sheet_by_index(0)
    print("Reading from sheet:", sheet.name)
    print("A total of", len(list(sheet.get_rows())), "rows")
    for row in list(sheet.get_rows())[1:]:
        row = Row(excel_row=row, read_before2006=True)
        l_row.append(row)
    return l_row


def in_leg(date, party):
    date = datetime.strptime(date, '%d-%m-%Y')
    l_legislaturas = get_legislaturas()
    for leg in l_legislaturas:
        if leg[1]<date<leg[2]:
            return leg[5] == party, leg[0], leg[5]

def get_legislaturas():

    l_legislaturas = [["5","29/06/1993","27/03/1996","9/07/1993","Felipe Gonzalez","PSOE"],
                      ["6","27/03/1996","05/04/2000","4/05/1996","José María Aznar","PP"],
                      ["7","05/04/2000","02/04/2004","26/04/2004","José María Aznar","PP"],
                      ["8","02/04/2004","01/04/2008","15/04/2004","José Luís Rodríguez","PSOE"],
                      ["9","01/04/2008","13/12/2011","11/04/2008","José Luís Rodríguez","PSOE"],
                      ["10","13/12/2011","13/01/2016","20/12/2016","Mariano Rajoy","PP"],
                      ["11","13/01/2016","19/07/2016","No otorgada","Candidato: Pedro Sánchez","PSOE"],
                      ["12","19/07/2016","21/05/2019","29/10/2016","Mariano Rajoy","PP"],
                      ["13","21/05/2019","03/12/2019","No otorgada","Candidato: Pedro Sánchez",""],
                      ["14","03/12/2019","09/12/2021","07/01/2020","Pedro Sánchez","PSOE"]]
    for leg in l_legislaturas:
        leg[1] = datetime.strptime(leg[1], '%d/%m/%Y')
        leg[2] = datetime.strptime(leg[2], '%d/%m/%Y')
    return l_legislaturas

def to_immspeech():
    l_todos = load_todos("C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\dataset\SERVIPOLI\\todos.xlsx")
    l_original = load_before2006(
        "C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\dataset\SERVIPOLI\\before 2006.xlsx") + load_before2006(
        "C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\dataset\SERVIPOLI\\after 2007.xlsx") + load_before2006(
        "C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\dataset\SERVIPOLI\\2007.xlsx") + load_before2006(
        "C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\dataset\SERVIPOLI\\2006.xlsx")
    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook("C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\dataset\SERVIPOLI\\immspeech.xlsx")
    worksheet = workbook.add_worksheet()

    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': 1})
    worksheet.write(0, 0, 'ParlSpeechIndex', bold)
    worksheet.write(0, 1, 'Cod1', bold)
    worksheet.write(0, 2, 'Date', bold)
    worksheet.write(0, 3, 'Speaker', bold)
    worksheet.write(0, 4, 'Party', bold)
    worksheet.write(0, 5, 'Num words', bold)
    worksheet.write(0, 6, 'Speech', bold)
    worksheet.write(0, 7, 'A.1', bold)
    worksheet.write(0, 8, 'A.2', bold)
    worksheet.write(0, 9, 'A.3', bold)
    worksheet.write(0, 10, 'A.4', bold)
    worksheet.write(0, 11, 'A.5', bold)
    worksheet.write(0, 12, 'E.1', bold)
    worksheet.write(0, 13, 'E.2', bold)
    worksheet.write(0, 14, 'E.3', bold)
    worksheet.write(0, 15, 'In Gob', bold)
    worksheet.write(0, 16, 'Legis', bold)
    d_cod1_row = clusters(l_todos,f_att= lambda r:r.cod1)
    d_cod1_row_original = clusters(l_original,f_att= lambda r:r.cod1)
    r = 1
    for cod1,l_rows in d_cod1_row.items():
        try:
            worksheet.write(r, 0, int(float(l_rows[0].Column1)))
        except:
            pass
        worksheet.write(r, 1, cod1)
        if cod1 in d_cod1_row_original:
            worksheet.write(r, 2, d_cod1_row_original[cod1][0].date)
            worksheet.write(r, 3, d_cod1_row_original[cod1][0].speaker)
            worksheet.write(r, 4, d_cod1_row_original[cod1][0].party)
            worksheet.write(r, 5, len(d_cod1_row_original[cod1][0].speech.split(" ")))
            worksheet.write(r, 6, d_cod1_row_original[cod1][0].speech, bold)
        else:
            print(cod1,"not in original")
        d_cat_row = clusters(l_rows,f_att= lambda r:r.cat)
        for d in '12345':
            if d not in d_cat_row:
                d_cat_row[d]=[]
        worksheet.write(r, 7, len(d_cat_row['1']))
        worksheet.write(r, 8, len(d_cat_row['2']))
        worksheet.write(r, 9, len(d_cat_row['3']))
        worksheet.write(r, 10, len(d_cat_row['4']))
        worksheet.write(r, 11, len(d_cat_row['5']))
        d_eva_row = clusters(l_rows,f_att= lambda r:r.eva)
        for d in '123':
            if d not in d_eva_row:
                d_eva_row[d]=[]
        worksheet.write(r, 12, len(d_eva_row['1']))
        worksheet.write(r, 13, len(d_eva_row['2']))
        worksheet.write(r, 14, len(d_eva_row['3']))
        if cod1 in d_cod1_row_original:
            is_in_leg,leg, party_at_gov=in_leg(d_cod1_row_original[cod1][0].date,d_cod1_row_original[cod1][0].party)
            worksheet.write(r, 15, is_in_leg)
            worksheet.write(r, 16, leg+":"+party_at_gov)

        r+=1
    workbook.close()
    pass





def negative_immspeech():
    immspeech = load_immspeech("C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\dataset\SERVIPOLI\immspeech_BUENO.xlsx")
    cong = load_Corp_congreso_v2("C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\Corp_Congreso_V2.xlsx")
    workbook = xlsxwriter.Workbook("C:\\Users\junqu\Google Drive\doctorado\Bias\Stereotype\dataset\SERVIPOLI\\immspeechNOIMM.xlsx")
    worksheet = workbook.add_worksheet()
    d_cod_ = {}
    d_text = {}
    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': 1})
    worksheet.write(0, 0, 'ParlSpeechIndex', bold)
    worksheet.write(0, 1, 'Cod1', bold)
    worksheet.write(0, 2, 'Date', bold)
    worksheet.write(0, 3, 'Speaker', bold)
    worksheet.write(0, 4, 'Party', bold)
    worksheet.write(0, 5, 'Num words', bold)
    worksheet.write(0, 6, 'Speech', bold)
    r = 1
    for row in immspeech:
        # to search speeches from same year same speaker
        year = row.date.year
        cod1 = row.cod1
        cod1_ = 1
        for r_cong in cong:

            if year == r_cong.date.year and row.parlSpeechIndex!=r_cong.Column1:
                if row.speaker == r_cong.speaker:
                    if r_cong.Column1 in d_cod_ or r_cong.text in d_text:
                        continue
                    if not speech_contains_key(r_cong.text):
                        d_cod_[r_cong.Column1] = r_cong
                        d_text[r_cong.text] = r_cong
                        worksheet.write(r, 0, r_cong.Column1)
                        worksheet.write(r, 1, cod1+"_"+str(cod1_))
                        cod1_ += 1
                        worksheet.write(r, 2, r_cong.date.strftime("%d-%m-%Y"))
                        worksheet.write(r, 3, r_cong.speaker)
                        worksheet.write(r, 4, r_cong.party)
                        worksheet.write(r, 5, len(r_cong.text.split(" ")))
                        worksheet.write(r, 6, r_cong.text, bold)
                        r+=1
    workbook.close()

negative_immspeech()
