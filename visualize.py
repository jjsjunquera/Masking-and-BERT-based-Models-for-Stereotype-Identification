# -*- coding: utf-8 -*-
# @Author: Jie Yang
# @Date:   2019-03-29 16:10:23
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-04-12 09:56:12


## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.
import numpy as np

latex_special_token = ["!@#$%^&*()"]

def generate(l_words_attention, latex_file, color='red', rescale_value = False):
	def prerpoc(text_list, attention_list):
		assert(len(text_list) == len(attention_list))
		if rescale_value:
			attention_list = rescale(attention_list)
		word_num = len(text_list)
		text_list = clean_word(text_list)
		return word_num, text_list
	with open(latex_file,'w') as f:
		f.write(r'''\documentclass{article}
\usepackage[utf8]{inputenc}
\special{papersize=210mm,297mm}
\usepackage{color}
\usepackage{tcolorbox}
\usepackage{CJK}
\usepackage{adjustbox}
\tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
\begin{document}
\begin{CJK*}{UTF8}{gbsn}'''+'\n')
		string = ""
		for code, probab, text_list, attention_list, masked in l_words_attention:
			# print(code, probab, text_list, attention_list, masked )
			word_num, text_list = prerpoc(text_list, attention_list)
			maxi = max(attention_list)
			mini = min(attention_list)
			# print(maxi, attention_list )
			# raise
			string += r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{'''+"\n"
			string += " \\textbf{CODE}: "+code+" \n\n  \\textbf{PROB}: "+str(round(probab,5))+ " \n\n \\textbf{BETO}: \n\n"
			for idx in range(word_num):
				token = text_list[idx]
				
				space = " "
				if token.startswith("\\#\\#"):
					token = token.replace("\\#\\#","")
					space = ""
				token = token.replace("á","\\'a")
				token = token.replace("é","\\'e")
				token = token.replace("í","\\'i")
				token = token.replace("ó","\\'o")
				token = token.replace("ú","\\'u")
				token = token.replace("ñ","\\~n")
				string += space+"\\colorbox{%s!%s}{"%(color, (attention_list[idx]-mini)*100/(maxi-mini))+"\\strut " + token+"}"
			masked = masked.replace("á","\\'a")
			masked = masked.replace("é","\\'e")
			masked = masked.replace("í","\\'i")
			masked = masked.replace("ó","\\'o")
			masked = masked.replace("ú","\\'u")
			masked = masked.replace("ñ","\\~n")
			string += "\n\n { \\small \\textbf{Masking}: \n\n" + masked + " }}}"
			string += r'''			
			
				\colorbox{red!00}{\strut } 
				
				 \colorbox{red!00}{\strut } 
				 
				  \colorbox{red!00}{\strut }  
				  
				 \colorbox{red!00}{\strut }}
				 
				 '''
			string += "\n\n"

		string += "\n"
		f.write(string+'\n')
		f.write(r'''\end{CJK*}
\end{document}''')

def rescale(input_list):
	the_array = np.asarray(input_list)
	the_max = np.max(the_array)
	the_min = np.min(the_array)
	rescale = (the_array - the_min)*100/(the_max-the_min)
	return rescale.tolist()


def clean_word(word_list):
	new_word_list = []
	for word in word_list:
		for latex_sensitive in ["\\", "%", "&", "^", "#", "_",  "{", "}"]:
			if latex_sensitive in word:
				word = word.replace(latex_sensitive, '\\'+latex_sensitive)
		new_word_list.append(word)
	return new_word_list

def create_latex(l_words_attention, name):
	color = 'red'
	generate(l_words_attention, "latex/"+name+"_sample.py", color,rescale_value=False)
