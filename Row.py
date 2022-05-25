class Row:
    def __init__(self, excel_row, read_from_spss=False, read_from_label88=False, read_from_hyp=True):
        self.tax1_orig = None
        if read_from_hyp:
            self.cod = excel_row[0].value
            self.tax1 = excel_row[1].value
            self.sentence = excel_row[4].value.lower()
            if len(self.sentence)>50:
                self.sentence = self.sentence[:50]
                print("TAKING FIRST 50 CHARACTERS")
        else:
            if read_from_spss:
                self.from_spss(excel_row)
            elif read_from_label88:
                self.from_88(excel_row)
            else:
              self.cod1, self.cod2, self.corpus, self.date, self.year, self.speaker, self.party, self.tax1, self.tax2, \
              self.sentence, self.kw_sen, self.speech, self.used_keywords = "", "", "", "", "", "", "", "", "", "", "", "", ""
              self.cod1 = excel_row[0].value if isinstance(excel_row[0].value, str) else str(
                  round(excel_row[0].value))
              self.cod2 = excel_row[1].value if isinstance(excel_row[1].value, str) else str(
                  round(excel_row[1].value))
              if len(excel_row) >= 13:
                  self.corpus = excel_row[2].value
                  self.date = excel_row[3].value
                  self.year = excel_row[4].value
                  self.speaker = excel_row[5].value
                  self.party = excel_row[6].value
                  self.tax1 = excel_row[7].value if isinstance(excel_row[7].value, str) else str(
                      round(excel_row[7].value))

                  self.tax2 = excel_row[8].value if isinstance(excel_row[8].value, str) else str(
                      round(excel_row[8].value))
                  self.sentence = excel_row[9].value.lower()
                  self.kw_sen = excel_row[10].value
                  self.speech = excel_row[11].value
                  self.used_keywords = excel_row[12].value
              elif len(excel_row) == 5:
                  self.tax1 = excel_row[2].value if isinstance(excel_row[2].value, str) else str(
                      round(excel_row[2].value))
                  self.tax2 = excel_row[3].value if isinstance(excel_row[3].value, str) else str(
                      round(excel_row[3].value))
                  self.sentence = excel_row[4].value
            self.cod = self.cod1 + "-" + self.cod2


    def from_spss(self, excel_row):
        self.cod1 = excel_row[0].value if isinstance(excel_row[0].value, str) else str(
              round(excel_row[0].value))
        self.cod2 = excel_row[1].value if isinstance(excel_row[1].value, str) else str(
              round(excel_row[1].value))
        self.sentence = excel_row[2].value.lower()
        self.tax1 = excel_row[19].value if isinstance(excel_row[19].value, str) else str(
                  round(excel_row[19].value))
        self.tax1_orig = str(excel_row[3].value)        
        self.total_agreement = excel_row[31].value if isinstance(excel_row[31].value, str) else str(
                  round(excel_row[31].value))
        if int(self.total_agreement) < 3:
            self.tax1  = None

    def from_88(self, excel_row):
        self.cod1 = excel_row[0].value if isinstance(excel_row[0].value, str) else str(
              round(excel_row[0].value))
        self.cod2 = excel_row[1].value if isinstance(excel_row[1].value, str) else str(
              round(excel_row[1].value))
        self.sentence = excel_row[8].value.lower()
        self.tax1 = str(int(excel_row[7].value))
        if self.tax1 != "88":
            self.tax1 = None


    def month(self):
        nums = self.date.split('-')
        if len(nums) == 3:
            return int(nums[1])

    def same_codes(self, row):
        return self.cod1 == row.cod1 and self.cod2 == row.cod2
