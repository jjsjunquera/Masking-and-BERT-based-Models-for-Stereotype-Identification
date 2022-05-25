# labels_tax1 = "1 2 3 4".split(" ")
# labels_tax2 = "5 6 7 8 9 10".split(" ")


class TaskStereotype():
    def __init__(self, task_choice):
        if task_choice=="in_taxonomy":
            print("Task: Taxonomy vs Not Taxonomy")
            self.task_name ="Taxonomy(1) vs Not Taxonomy(0)"
            self.task = self.to_tax_notax
        elif task_choice == "pro_anti":
            print("Task: Pro vs Anti Immigration")
            self.task_name ="Pro(0) vs Anti (1) Immgration"
            self.task = self.to_pro_anti
        elif task_choice == "t_f":
            print("Task: Truthful vs Deception")
            self.task_name ="Truthful (0) vs Deception (1)"
            self.task = self.to_t_d
        elif task_choice == "hyp_nohyp":
            print("Task: Hyp(0) vs NonHyp (1)")
            self.task_name ="Hyp(0) vs NonHyp (1)"
            self.task = self.to_hyp_nonhyp
        elif task_choice == "l_main_r":
            print("Task: Left(0) vs Main(1) vs Right(2)")
            self.task_name ="Left(0) vs Main(1) vs Right(2)"
            self.task = self.to_l_main_r
        else:
            self.task_name ="Original tags"
            self.task = self.original_tags

    def to_label(self, label, cod):
        return self.task(label)

    def original_tags(self, label):
        return label

    def to_tax_notax(self,label):
        if label == None:
            return None
        if "88" in label:
            return 0
        if label in "1 2 3 4 5".split(" "):
          return 1

    def to_hyp_nonhyp(self,label):
        if label == "mainstream":
            return 1
        elif label in ("left","right"):
            return 0
        else:
            return None

    def to_t_d(self,label):
        if label == False:
            return 0
        elif label == True:
            return 1
        else:
            return None

    def to_l_main_r(self,label):
        if label == "left":
            return 0
        if label == "mainstream":
            return 1
        elif label == "right":
            return 2
        else:
            return None

    def to_pro_anti(self,label):
        if label == None:
          return None
        if label in "1 2".split(" "):
          return 0

        if label in "4 5".split(" "):
          return 1

        if label =="3":
          return None
        if label =="88":
          return None

        raise("bad label")



