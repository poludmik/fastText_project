import numpy as np
from scipy import spatial
import nltk
import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk import word_tokenize


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


def pd_df_to_train_test_sets(df, test_percentage=0.1):
    all_testing_questions = []
    for index, row in df.iterrows():
        all_testing_questions.append((int(index), row['question'], row['class']))
    if test_percentage == 1:
        x_train = None
        x_test = np.array(all_testing_questions)
    else:
        x_train, x_test = train_test_split(np.array(all_testing_questions), test_size=test_percentage)
    return x_train, x_test


def classify_one_sentence(df, m, sentence, idx=None):
    class_of_max_similarity = 0
    question_max = ""
    max_similarity = -np.inf
    for index, row in df.iterrows():
        if index == idx: # if test question was from the same train dataframe
            continue
        
        similarity = cosine_sim_scipy(m.get_sentence_vector(row['question']), m.get_sentence_vector(sentence))
        # similarity = cosine_sim_scipy(get_average_vector(m, row['question']), get_average_vector(m, sentence))

        if similarity > max_similarity:
            max_similarity = similarity
            question_max = row['question']
            class_of_max_similarity = row['class']
    return class_of_max_similarity, question_max


def get_classification_error(df, m, x_test):
    number_right = 0
    for idx, question, true_class in x_test:
        classified_class, max_similar_question = classify_one_sentence(df, m, question, idx=int(idx))
        if classified_class == int(true_class):
            number_right += 1
    return 1.0 - number_right / x_test.shape[0]


def get_class2questions_dict(df):
    class2question = {}
    for index, row in df.iterrows():
        if row['class'] not in class2question:
            class2question[row['class']] = []
        class2question[row['class']].append(row['question'])
    return class2question


def classify_one_sentence_based_on_average(m, class2question_dict, sentence):
    class_of_max_similarity = 0
    max_similarity = -np.inf
    v_shape = get_average_vector(m, "dummy").shape
    for class_id in class2question_dict:
        vectors_in_tr_data = np.zeros(v_shape)
        c = 0
        for q in class2question_dict[class_id]:
            if q != sentence:
                vectors_in_tr_data = vectors_in_tr_data + m.get_sentence_vector(q)
                c += 1
        average_tr_vector = vectors_in_tr_data / c

        similarity = cosine_sim_scipy(average_tr_vector, m.get_sentence_vector(sentence))
        if similarity > max_similarity:
            max_similarity = similarity
            class_of_max_similarity = class_id
    return class_of_max_similarity


def get_classification_error_with_averages(class2questions_dict, m, x_test):
    number_right = 0
    for idx, question, true_class in x_test:
        classified_class = classify_one_sentence_based_on_average(m, class2questions_dict, question)
        if classified_class == int(true_class):
            number_right += 1
    return 1.0 - number_right / x_test.shape[0]


