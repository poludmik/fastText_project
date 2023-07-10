"""
Adapted from Adam Jirkovsky - jirkoada.
"""
import pandas as pd
import json
import operator
from collections import Counter

from cs_lemmatizer import *


# nltk.download('wordnet')
# lmtzr = WordNetLemmatizer()
# stem = PorterStemmer()
# stem_method = lmtzr.lemmatize



def remove_tags():
    ans_df = pd.read_excel("Q78_answers.xlsx")
    ans_df["answer"] = ans_df["answer"].str.replace(r'<sub alias="([^"]*)">[^<]*</sub>', r'\1', regex=True)
    ans_df.to_excel("Q78_answers_no_tags.xlsx")


def count_word_probs_in_answers(path_to_corpus, path_to_save):
    ans_df = pd.read_excel(path_to_corpus)
    corpus = ans_df['answer'].str.cat(sep="\n")
    # words = corpus.lower().split()

    words = LMTZR.tokenize(corpus)
    words = [LMTZR.lemmatize_cs(w) for w in words]
    
    probs = {}
    for word in words:
        word = word.strip('.')
        if word in probs.keys():
            probs[word] += 1
        else:
            probs[word] = 1
    for key in probs.keys():
        probs[key] /= len(words)
    with open(path_to_save, "w") as f:
        json.dump(probs, f)


def count_word_probs_in_questions(path_to_questions, path_to_save):
    ans_df = pd.read_excel(path_to_questions)
    corpus = ans_df['question'].str.cat(sep="\n")
    # words = corpus.lower().split()
    words = LMTZR.tokenize(corpus)
    words = [LMTZR.lemmatize_cs(w) for w in words]
    
    probs = {}
    for word in words:
        word = word.strip('.')
        if word in probs.keys():
            probs[word] += 1
        else:
            probs[word] = 1
    for key in probs.keys():
        probs[key] /= len(words)
    with open(path_to_save, "w") as f:
        json.dump(probs, f)


def combine_dicts(a, b, save_path=None):
    # return dict(a.items() + b.items() +
    #     [(k, operator.add(a[k], b[k])) for k in set(b) & set(a)])
    combined_dict = Counter(a) + Counter(b)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(combined_dict, f)
    return combined_dict


if __name__ == "__main__":
    #remove_tags()
    # count_word_probs_in_answers()
    # count_word_probs_in_questions()
    pass