import numpy as np
from scipy import spatial
import nltk
import fasttext
import pandas as pd

from nltk import word_tokenize
nltk.download('punkt')



def cosine_sim_scipy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return 1 - spatial.distance.cosine(vec1, vec2)


def cosine_sim_numpy(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return (vec1 @ vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def tokenize_sentence(sentence: str, use_nltk=False) -> list[str]:
    if use_nltk:
        return word_tokenize(sentence)

    return sentence.rstrip().lower().split()


def get_average_vector(m: fasttext.FastText, sentence: str) -> np.ndarray:
    tokens = tokenize_sentence(sentence, use_nltk=False)
    return np.average([m.get_word_vector(t) for t in tokens], axis=0)


def classify_one_sentence(df, m, sentence):
    class_of_max_similarity = 0
    question_max = ""
    max_similarity = -np.inf
    for index, row in df.iterrows():
        similarity = cosine_sim_scipy(get_average_vector(m, row['question']), get_average_vector(m, sentence))
        if similarity > max_similarity:
            max_similarity = similarity
            question_max = row['question']
            class_of_max_similarity = row['class']
    return class_of_max_similarity, question_max



