from Row import Row


class CreateCorpus:
    @staticmethod
    def filtering_rows(source_files, labeled_files, f_filter, name_new_file):
        rows = CreateCorpus.load_rows(source_files)
        #rows_with_labels = CreateCorpus.load_rows(labeled_files)
        rows_with_labels = CreateCorpus.load_rows_from_sheet(labeled_files)
        filtered_rows = list(sorted([row for row in rows if f_filter(row)], key=lambda row: int(row.year)))
        for labeled_row in rows_with_labels:
            for filtered_row in filtered_rows:
                if filtered_row.same_codes(labeled_row):
                    filtered_row.tax1, filtered_row.tax2 = labeled_row.tax1, labeled_row.tax2
                    break

        d_speech = clusters(filtered_rows, f_att=lambda x:x.cod1)
        filtered_rows = []
        for cod1, l_row in d_speech.items():
            kws = l_row[0].used_keywords.split("; ")
            if len([kw_imm for kw_imm in kws if kw_imm.startswith("inmigr")])==0:
                continue

            cant_sentences_kw = len([row for row in l_row if len(row.kw_sen)>3])
            if cant_sentences_kw/len(l_row) <= 0.2:
                continue

            filtered_rows.extend(l_row)
        CreateCorpus.create_file(filtered_rows, name_new_file)

    @staticmethod
    def load_rows(source_files, label88=False):
        import xlrd
        l_result = []
        for fil_name in source_files:
            wb = xlrd.open_workbook(fil_name)
            sheet = wb.sheet_by_index(0)
            print("Reading from sheet:",sheet.name)
            print("A total of",len(list(sheet.get_rows())), "rows")
            index = 0
            for row in sheet.get_rows():
              if not isinstance(row[0].value, str):
                if isinstance(row[0].value, float):
                  row[0].value = str(int(row[0].value))

                  l_result.append(Row(excel_row=row,read_from_label88=label88))
                else:
                  print(row)
              elif  not row[0].value.startswith("COD"):
                l_result.append(Row(excel_row=row,read_from_label88=label88))
              # else:
              #     print(row)
              index=index+1
        print()
        return l_result


    @staticmethod
    def load_rows_from_spss(source_files):
        import xlrd
        l_result = []
        for fil_name in source_files:

            wb = xlrd.open_workbook(fil_name)

            sheet = wb.sheet_by_index(0)
            print("Reading from sheet:",sheet.name)
            print("A total of",len(list(sheet.get_rows())), "rows")
            index = 0
            for row in sheet.get_rows():
              if not isinstance(row[0].value, str):
                if isinstance(row[0].value, float):
                  row[0].value = str(int(row[0].value))

                  l_result.append(Row(excel_row=row, read_from_spss=True))
                else:
                  print("see row[0].value", row)
              elif  not row[0].value.startswith("COD"):
                l_result.append(Row(excel_row=row, read_from_spss=True))
              # else:
              #     print(row)
              index=index+1
        print()

        return l_result
    
    @staticmethod
    def load_rows_from_hyp(source_files):
        import xlrd
        l_result = []
        for fil_name in source_files:
            wb = xlrd.open_workbook(fil_name)
            sheet = wb.sheet_by_index(0)
            print("Reading from sheet:",sheet.name)
            print("A total of",len(list(sheet.get_rows())), "rows")
            for row in sheet.get_rows():
                l_result.append(Row(excel_row=row, read_from_hyp=True))

        l_result = l_result[1:]
        return l_result

