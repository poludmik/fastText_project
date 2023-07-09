"""
Adapted from Adam Jirkovsky - jirkoada.
"""
import pandas as pd
import json

import nltk
from cs_lemmatizer import *

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('wordnet')
# lmtzr = WordNetLemmatizer()
# stem = PorterStemmer()
# stem_method = lmtzr.lemmatize



def remove_tags():
    ans_df = pd.read_excel("Q78_answers.xlsx")
    ans_df["answer"] = ans_df["answer"].str.replace(r'<sub alias="([^"]*)">[^<]*</sub>', r'\1', regex=True)
    ans_df.to_excel("Q78_answers_no_tags.xlsx")


def count_word_probs_in_answers():
    ans_df = pd.read_excel("upv_faq/Q78_answers_no_tags.xlsx")
    corpus = ans_df['answer'].str.cat(sep="\n")
    words = corpus.lower().split()

    # words = LMTZR.tokenize(corpus)
    # words = [LMTZR.lemmatize_cs(w) for w in words]
    
    probs = {}
    for word in words:
        word = word.strip('.')
        if word in probs.keys():
            probs[word] += 1
        else:
            probs[word] = 1
    for key in probs.keys():
        probs[key] /= len(words)
    #print(probs)
    with open("upv_faq/Q78_answer_word_probs.json", "w") as f:
        json.dump(probs, f)


def count_word_probs_in_questions():
    ans_df = pd.read_excel("upv_faq/Q78_questions.xlsx")
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
    # print(probs)
    with open("upv_faq/Q78_question_word_probs.json", "w") as f:
        json.dump(probs, f)


if __name__ == "__main__":
    #remove_tags()
    # count_word_probs_in_answers()
    # count_word_probs_in_questions()
    pass