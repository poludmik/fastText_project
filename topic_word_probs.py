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

from keybert import KeyBERT



def remove_tags():
    ans_df = pd.read_excel("Q78_answers.xlsx")
    ans_df["answer"] = ans_df["answer"].str.replace(r'<sub alias="([^"]*)">[^<]*</sub>', r'\1', regex=True)
    ans_df.to_excel("Q78_answers_no_tags.xlsx")


def count_word_probs_in_corpuses(path_to_save=None, 
                                 path_to_questions=None, 
                                 path_to_answers=None, 
                                 tokenizer=LMTZR.clean_corpus):
    # LMTZR.clean_corpus has sw=lm=True as default argument values.
    words = []
    if path_to_answers:
        ans_df = pd.read_excel(path_to_answers)
        corpus = (ans_df['answer'].astype(str)).str.cat(sep="\n")
        words += tokenizer(corpus)    
    if path_to_questions:
        q_df = pd.read_excel(path_to_questions)
        corpus = (q_df['question'].astype(str)).str.cat(sep="\n")
        words += tokenizer(corpus)
    
    probs = {}
    for word in words:
        word2 = word.strip('.')
        if word2 in probs.keys():
            probs[word2] += 1
        else:
            probs[word2] = 1

    keys_to_remove = []
    for key in probs.keys():
        # if probs[key] < 5:
        #     keys_to_remove.append(key)
        probs[key] /= len(words)
    for k in keys_to_remove:
        probs.pop(k)

    if path_to_save:
        with open(path_to_save, "w") as f:
            json.dump(probs, f)

    return probs, len(words)


def combine_dicts(a, b, save_path=None):
    combined_dict = Counter(a) + Counter(b)
    if save_path:
        with open(save_path, "w") as f:
            json.dump(combined_dict, f)
    return combined_dict


def general_to_corpus_probs(dict1_c, dict2_g):
    new_dict = {}
    filter_words = ['"']
    for key in dict1_c.keys():
        if key in filter_words:
            continue
        if key in dict2_g:
            new_dict[key] = dict2_g[key] / dict1_c[key]
        else:
            new_dict[key] = dict1_c[key]

    max_value = max(new_dict.values())
    for key in new_dict.keys():
        new_dict[key] /= max_value # to normalize to [0, 1]

    return new_dict


def get_TFIDF_threshold_probabilities(tfidf_matrix, feature_names):
    """
    For each word, take the maximum value from tf-idf matrix across all classes.
    If a word is important in some class -> it's p_d value will be big.
    """
    max_in_all_classes = np.squeeze(np.max(tfidf_matrix.toarray(), axis=0))
    p_d = {}
    for i, w in enumerate(feature_names):
        # if max_in_all_classes[i] < 0.1:
            # continue
        p_d[w] = max_in_all_classes[i]
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


def get_weights_with_KeyBERT(structured_q: list[str], sw=False, lm=False, path_to_save=None):
    weights = {}
    kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
    for c_str in structured_q:
        tmp_c = " ".join(LMTZR.clean_corpus(c_str, rm_stop_words=sw, lemm=lm))
        keywords = kw_model.extract_keywords(tmp_c, keyphrase_ngram_range=(1, 2), 
                                     use_mmr=True, 
                                     diversity=0.5,
                                     top_n=3,
                                     )
        for s, p in keywords:
            if p <= 0:
                continue
            tmp_s = LMTZR.clean_corpus(s) # lm and sw aready cleaned up
            for w in tmp_s:
                if w not in weights:
                    weights[w] = p
                else:
                    if p > weights[w]:
                        weights[w] = p

    if path_to_save:
        with open(path_to_save, "w") as f:
            json.dump(weights, f)
                
    return weights




if __name__ == "__main__":
    # remove_tags()
    # count_word_probs_in_answers()
    # count_word_probs_in_questions()
    pass