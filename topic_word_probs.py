"""
Adapted from Adam Jirkovsky - jirkoada.
"""
import pandas as pd
import json
import operator
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

from cs_lemmatizer import *


def remove_tags():
    ans_df = pd.read_excel("Q78_answers.xlsx")
    ans_df["answer"] = ans_df["answer"].str.replace(r'<sub alias="([^"]*)">[^<]*</sub>', r'\1', regex=True)
    ans_df.to_excel("Q78_answers_no_tags.xlsx")


def count_word_probs_in_corpuses(path_to_save=None, path_to_questions=None, path_to_answers=None):
    words = []
    if path_to_answers:
        ans_df = pd.read_excel(path_to_answers)
        corpus = ans_df['answer'].str.cat(sep="\n")
        words += LMTZR.clean_corpus(corpus)    
    if path_to_questions:
        q_df = pd.read_excel(path_to_questions)
        corpus = q_df['question'].str.cat(sep="\n")
        words += LMTZR.clean_corpus(corpus)
    
    probs = {}
    for word in words:
        word = word.strip('.')
        if word in probs.keys():
            probs[word] += 1
        else:
            probs[word] = 1

    keys_to_remove = []
    for key in probs.keys():
        if probs[key] < 5:
            keys_to_remove.append(key)
        probs[key] /= len(words)
    for k in keys_to_remove:
        probs.pop(k)

    if path_to_save:
        with open(path_to_save, "w") as f:
            json.dump(probs, f)

    return probs, len(words)


def combine_dicts(a, b, save_path=None):
    # return dict(a.items() + b.items() +
    #     [(k, operator.add(a[k], b[k])) for k in set(b) & set(a)])
    combined_dict = Counter(a) + Counter(b)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(combined_dict, f)
    return combined_dict


def get_TFIDF_threshold_probabilities(tfidf_matrix, feature_names):

    max_in_all_classes = np.squeeze(np.max(tfidf_matrix, axis=0).toarray())
    p_d = {}
    print(np.mean(max_in_all_classes))
    for i, w in enumerate(feature_names):
        p_d[w] = max_in_all_classes[i]

    print("n words =", len(p_d))
    print(" \"probs\" from tf_idf:", dict(sorted(p_d.items(), key=lambda item: -item[1])))

    return p_d


def histogram_of_words(dict_of_probs, number_of_words):
    """
    Outputs every second word in a dict
    """
    i = 0
    dict_of_probs = dict(sorted(dict_of_probs.items(), key=lambda item: item[1]))
    delete_keys = []
    for key in dict_of_probs.keys():
        dict_of_probs[key] *= number_of_words
        i += 1
        if i % 2 != 0:
            delete_keys.append(key)

    for k in delete_keys:
        dict_of_probs.pop(k)

    print(dict(sorted(dict_of_probs.items(), key=lambda item: -item[1])))
    print("Number of distinctive words =", len(dict_of_probs))
    plt.figure(figsize=(10, 15), dpi=80)
    plt.xticks(fontsize=9)
    plt.barh(list(dict_of_probs.keys()), dict_of_probs.values(), color='g')
    plt.savefig('780_upv_questions/word_counts.png')




if __name__ == "__main__":
    #remove_tags()
    # count_word_probs_in_answers()
    # count_word_probs_in_questions()
    pass