import fasttext
import numpy as np
import pandas as pd
import warnings
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import os
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet')
lmtzr = WordNetLemmatizer()
stem = PorterStemmer()


class TFIDF_Classifier:
    def __init__(self, questions_path):
        self.questions_path = questions_path
        self.structured_list = None
        self.TFIDF_matrix = None
        self.feature_names = None
        self.stem_method = lmtzr.lemmatize
        # self.stem_method = stem.stem

    def structure_data(self, test_data_percent=None):
        i = 0
        df = pd.read_excel(self.questions_path)

        n_of_test = int(test_data_percent * len(df.index))
        drop_indices = np.random.choice(df.index, n_of_test, replace=False)
        df_tests = df.iloc[drop_indices]
        
        if test_data_percent == 1:
            df_subset = df
        else:
            df_subset = df.drop(drop_indices)

        self.structured_list = ["" for _ in range(df['class'].max() + 1)]

        for index, row in df_subset.iterrows():
            lemmatized = [self.stem_method(word) for word in word_tokenize(row['question'])]
            self.structured_list[int(row['class'])] += " " + ' '.join(lemmatized)

        test_data = []
        for index, row in df_tests.iterrows():
            test_data.append((row['question'], row['class']))

        return test_data

    def get_TFIDF_matrix(self):
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(self.structured_list)
        self.feature_names = vectorizer.get_feature_names_out()
        self.TFIDF_matrix = X
        print("TFIDF matrix shape:", X.shape)
    
    def classify_sentence(self, sentence):
        max_p = -np.inf
        classified_c = None
        for c in range(self.TFIDF_matrix.shape[0]): # classes
            p = 0
            for word in word_tokenize(sentence):
                w = self.stem_method(word)
                if w not in self.feature_names:
                    continue
                p += self.TFIDF_matrix[c, np.where(self.feature_names == w)[0][0]]
            p /= len(word_tokenize(sentence))
            if p > max_p:
                max_p = p
                classified_c = c
        # print("C:", classified_c)
        # print(max_p)
        return classified_c

    def classify_test_sentences_list(self, test_list):
        right_n = 0
        for s, true_c in test_list:
            c = self.classify_sentence(s)
            if c == true_c:
                right_n += 1
        print("Got right:", right_n / len(test_list))


