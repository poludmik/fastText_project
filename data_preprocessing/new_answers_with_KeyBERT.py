from keybert import KeyBERT
import pandas as pd
import sys
import os
 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from cs_lemmatizer import LMTZR


def get_keyword_answers(path_to_a): # sucks
    
    df = pd.read_excel(path_to_a)

    kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")

    modified_rows = []
    classes = []
    already_have = set()
    for index, row in df.iterrows():
        if int(row['class']) in already_have:
            continue
        already_have |= {int(row['class'])}
        classes.append(int(row['class']))
        tmp_c = " ".join(LMTZR.clean_corpus(row['answer'], rm_stop_words=False, lemm=False))
        keywords = kw_model.extract_keywords(tmp_c, keyphrase_ngram_range=(1, 3), 
                                       use_mmr=True, 
                                       diversity=0.5,
                                       top_n=5,
                                       )
        new_row = " ".join([w[0] for w in keywords if w[1] > 0])
        modified_rows.append(new_row)

    df = pd.DataFrame({"answer": modified_rows, "class": classes})
    df.to_excel("data_preprocessing/keyword_answers.xlsx")


if __name__ == "__main__":
    path_to_a = "upv_faq/data/FAQ76_answers.xlsx"
    get_keyword_answers(path_to_a=path_to_a)
