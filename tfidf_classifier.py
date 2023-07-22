import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

# nltk.download('wordnet')
# lmtzr = WordNetLemmatizer()
# stem = PorterStemmer()

import simplemma
from simplemma import simple_tokenizer
from cs_lemmatizer import *


class TFIDF_Classifier:
    def __init__(self, questions_path, rm_sw=True, lm=True):
        self.questions_path = questions_path
        self.structured_list = None
        self.TFIDF_matrix = None
        self.feature_names = None
        self.rm_sw = rm_sw
        self.lm = lm

    def leave_one_out_test(self):
        df = pd.read_excel(self.questions_path)
        n_got_right = 0
        for index, row in tqdm(df.iterrows()):
            sentence = row['question']
            true_class = int(row['class'])

            self.structured_list = ["" for _ in range(df['class'].max() + 1)]
            for index2, row2 in df.iterrows():
                if index == index2:
                    continue
                words = LMTZR.clean_corpus(row2['question'], self.rm_sw, self.lm)
                self.structured_list[int(row2['class'])] += " " + ' '.join(words)
            self.get_TFIDF_matrix()
            if self.classify_sentence(sentence) == true_class:
                n_got_right += 1
            
        return round(n_got_right / len(df.index), 3)


    def structure_data(self, test_data_percent=None, sents_idxs_to_leaveout=[], tokenizer=None):
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
            if index in sents_idxs_to_leaveout: # for leave-one-out mean match test
                continue
            if tokenizer is None:
                lemmatized = LMTZR.clean_corpus(row['question'], self.rm_sw, self.lm)
            else:
                lemmatized = tokenizer(row['question'])
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
        # print("TFIDF matrix shape:", X.shape)
        return X, self.feature_names

    def classify_sentence(self, sentence):
        max_p = -np.inf
        classified_c = None
        for c in range(self.TFIDF_matrix.shape[0]): # classes
            p = 0
            words = LMTZR.clean_corpus(sentence, self.rm_sw, self.lm)
            for word in words:
                w = self.lemmatize_cs_w(word)
                if w not in self.feature_names:
                    continue
                p += self.TFIDF_matrix[c, np.where(self.feature_names == w)[0][0]]
            p /= len(words)
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


