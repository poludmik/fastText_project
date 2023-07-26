"""
Adapted from Adam Jirkovsky - jirkoada.
"""
import fasttext
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import json
import os

from cs_lemmatizer import *
from tfidf_classifier import *
from topic_word_probs import *
from bert_like_models.slavic_bert_tests import SlavicBERT


def extract_word_probs(model_path: str, corpus_size: int = 4.1e9):
    """
    Uses word occurency counts included in FT model and training corpus size to create a dict of 
    word frequencies/probabilities and saves it to a json file inside current working directory
    - Only uncompressed models are supported
    """
    model = fasttext.load_model(model_path)
    words, freqs = model.get_words(include_freq=True)
    probs = list(map(lambda x: float(x) / corpus_size, freqs))
    word_probs = dict(zip(words, probs))
    probs_path = os.path.splitext(os.path.basename(model_path))[0] + "_probs.json"
    with open(probs_path, "w") as out:
        json.dump(word_probs, out)
    return probs_path


class FAQ:
    def __init__(
            self, 
            model, 
            questions_path, 
            answers_path=None,
            probs=None, 
            alpha=1e-4, 
            compressed=False,
            rm_stop_words=False,
            lemm=False,
            tfidf_weighting=False,
            slBert=False
        ):
        self.model = model
        self.answers = None
        self.sentence_embedding = self.mean_sentence_embedding
        self.word_probs = probs
        self.alpha = alpha
        self.rm_stop_words = rm_stop_words
        self.lemm = lemm
        self.path_to_q = questions_path
        self.tfidf_weighting = tfidf_weighting
        self.slBert = slBert

        if compressed:
            self.get_w_vec = self.model.word_vec
        elif slBert:
            self.get_w_vec = None
            self.sentence_embedding = self.SlavicBERT_mean_sentence_embedding
        else:
            self.get_w_vec = self.model.get_word_vector

        if questions_path.split(".")[1] == "xlsx":
            
            self.questions = pd.read_excel(questions_path)
        elif questions_path.split(".")[1] == "csv":
            self.questions = pd.read_csv(questions_path, sep="\t")
        else:
            raise "Unsupported data file"
        
        if answers_path and questions_path.split(".")[1] == "xlsx":
            self.answers = pd.read_excel(answers_path)
        elif answers_path and questions_path.split(".")[1] == "csv":
            self.answers = pd.read_csv(answers_path, sep="\t")
        elif answers_path:
            raise "Unsupported data file"

        if alpha is not None and probs is not None:
            self.sentence_embedding = self.weighted_sentence_embedding    

        # Create embedding database - matrix of embedding vectors for each question
        self.db = np.array([self.sentence_embedding(q) for q in self.questions["question"]])

        # Mean embedding database - holds averages of question embeddings for each class
        self.mean_db = np.zeros([self.questions["class"].nunique(), self.db.shape[1]])
        for i, cls in enumerate(self.questions["class"].unique()):
            imin = self.questions[self.questions["class"] == cls].index.min()
            imax = self.questions[self.questions["class"] == cls].index.max()
            self.mean_db[i, :] = self.db[imin:imax+1, :].mean(axis=0)

        # Answer database - matrix of embeddings for each unique answer
        if self.answers is not None:
            self.ans_db = np.array([self.sentence_embedding(a) for a in self.answers['answer']])

    def default_sentence_embedding(self, sentence):
        # Unsuported by compressed models, may be removed
        embedding = self.model.get_sentence_vector(sentence.lower().replace('\n', ' '))
        return embedding/np.linalg.norm(embedding)
    
    def SlavicBERT_mean_sentence_embedding(self, sentence):
        return self.model.get_mean_sentence_embedding(sentence, sw=self.rm_stop_words, lm=self.lemm)

    def mean_sentence_embedding(self, sentence):
        # Same as default, but computed manually

        words = LMTZR.clean_corpus(sentence, rm_stop_words=self.rm_stop_words, lemm=self.lemm)

        wes = np.array([self.get_w_vec(w) for w in words])
        wes /= np.linalg.norm(wes, axis=1)[:, np.newaxis] + 1e-9
        se = np.mean(wes, axis=0)
        return se/np.linalg.norm(se)

    def weighted_sentence_embedding(self, sentence):
        # Computes weighted sentence embedding acoording to: https://openreview.net/pdf?id=SyK00v5xx
        
        def word_probability(word):
            if word in self.word_probs.keys():
                return self.word_probs[word]
            return min(self.word_probs.items(), key=lambda x: x[1])[1]
            # return 0 # also works, but assigning some number seems to work better

        if self.slBert:
            words = self.model.tokenizer.tokenize(sentence)
            wes = self.model.get_mean_sentence_embedding(sentence, mean=False)
        else:
            words = LMTZR.clean_corpus(sentence, rm_stop_words=self.rm_stop_words, lemm=self.lemm)
            wes = np.array([self.get_w_vec(w) for w in words])
        
        probs = np.array([word_probability(w) for w in words])[:, np.newaxis]
        wes /= np.linalg.norm(wes, axis=1)[:, np.newaxis] + 1e-9

        wes *= 1 / (self.alpha + probs)
        if self.tfidf_weighting:
            wes *= probs

        se = np.mean(wes, axis=0)
        return se / (np.linalg.norm(se) + 1e-9)

    def total_confusion(self):
        # Shows a heatmap of cosine similarities of all question pairs
        # Regions within the same class are enclosed in red squares
        # Click a pixel to print out aditional info about the matching question pair
        cm = self.db @ self.db.T
        am = np.argmax(cm, axis=1)
        # for i in range(am.shape[0]):
        #     if am[i] != i:
        #         print("Ambiguous match:")
        #         print(self.questions["question"][i], i, self.questions["class"][i])
        #         print(self.questions["question"][am[i]], am[i], self.questions["class"][am[i]])
        #         print()

        def onclick(event):
            if event.xdata is None or event.ydata is None:
                return
            x = int(np.round(event.xdata))
            y = int(np.round(event.ydata))
            print(f"Pixel coords: {x}, {y}")
            print(f"Q1: {self.questions['question'][x]}, Class {self.questions['class'][x]}")
            print(f"Q2: {self.questions['question'][y]}, Class {self.questions['class'][y]}")
            print(f'Similarity: {cm[x, y]}')
            print()

        fig = plt.figure(figsize=(10, 7))
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.matshow(cm, 0)
        for cls in self.questions["class"].unique():
            ul = self.questions[self.questions["class"] == cls].index.min() - 0.5
            edge = self.questions[self.questions["class"] == cls].shape[0]
            plt.gca().add_patch(Rectangle((ul, ul), edge, edge, linewidth=1, edgecolor='r', facecolor='none'))
        plt.title("Confusion matrix for all question matches")
        plt.show()

    def mean_match_test(self, verb=False, show_cm=False, show_time=2.0):
        # Determines question class by comparing it with mean database and computes classification accuracy
        cm = self.db @ self.mean_db.T
        am = np.argmax(cm, axis=1)
        preds = am
        gts = self.questions["class"].to_numpy(dtype=int)
        hits = preds == gts
        acc = hits.mean()

        sorted_indexes = np.argsort(cm, axis=1)
        preds_second = sorted_indexes[:, -2]
        hits_second = (preds_second == gts) * (hits == False)
        acc_second = hits_second.mean()

        #print(f"Mean match accuracy: {acc}")

        if not hits.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(hits):
                if not b:
                    print(f"{self.questions['question'][i]} : Class {am[i]}")

        if show_cm:
            cm = confusion_matrix(gts, preds)
            fig = plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=False)
            plt.title("Mean matching confusion matrix")
            plt.xlabel("Prediction")
            plt.ylabel("True class")
            plt.draw()
            plt.pause(show_time)
            return acc, fig
        return round(acc, 3), round(acc_second, 3)
    
    def cross_match_test(self, verb=False, show_cm=False, show_time=2.0):
        # Computes cosine similarities of all question pairs
        # A question is succesfully matched, if its second highest similarity is with a question of the same class
        # Computes accuracy as the ratio of succesfull matches
        cm = self.db @ self.db.T

        # Would be better with np.fill_diagonal, but due to ambiguous questions, 
        # some might have similarity bigger with another question; shouldn't really matter.
        # np.fill_diagonal(cm, np.inf) 

        am = np.argsort(cm, axis=1)[:, -2]
        cls_ids = self.questions["class"].to_numpy(dtype=int)
        hits = cls_ids == cls_ids[am]
        acc = hits.mean()

        # Second nearest neighbour
        am2 = np.argsort(cm, axis=1)[:, -3]
        hits2 = (cls_ids == cls_ids[am2]) * (hits == False)
        acc2 = hits2.mean()

        if not hits.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(hits):
                if not b:
                    print(f"{self.questions['question'][i]} : {self.questions['question'][am[i]]}")

        if show_cm:
            cm = confusion_matrix(cls_ids, cls_ids[am])
            fig = plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=False)
            plt.title("Question cross-matching confusion matrix")
            plt.xlabel("Prediction")
            plt.ylabel("True class")
            plt.draw()
            plt.pause(show_time)
            return acc, fig
        return round(acc, 3), round(acc2, 3)

    def ans_test(self, verb=False, show_cm=False, show_time=2.0):
        # Classifies questions by directly comparing them with embedded answers and computes accuracy
        if self.answers is None:
            warnings.warn("Answers are not available")
            return None
        cm = self.db @ self.ans_db.T
        am = np.argmax(cm, axis=1)
        preds = self.answers["class"].to_numpy(dtype=int)[am]
        gts = self.questions["class"].to_numpy(dtype=int)
        hits = preds == gts

        acc = hits.mean()
        #print(f"Answer match accuracy: {acc}")

        if not hits.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(hits):
                if not b:
                    print(f"{self.questions['question'][i]} : {self.answers['answer'][am[i]]}")

        if show_cm:
            cm = confusion_matrix(gts, preds)
            fig = plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=True)
            plt.title("Answer matching confusion matrix")
            plt.xlabel("Prediction")
            plt.ylabel("True class")
            plt.draw()
            plt.pause(show_time)
            return acc, fig
        return acc, None

    def identify(self, question):
        v = self.sentence_embedding(question)
        sims = self.db @ v[:, np.newaxis]
        return np.argmax(sims)
    
    def identify_direct_answer(self, question):
        v = self.sentence_embedding(question)
        a_sims = self.ans_db @ v[:, np.newaxis]
        return np.argmax(a_sims)
    
    def match(self, question):
        matched_q = self.questions['question'][self.identify(question)]
        #print(f"Matched question: {matched_q}")
        return matched_q, self.questions['class'][self.identify(question)]

    def answer(self, question):
        if self.answers is None:
            warnings.warn("Answers are not available")
            return None
        ans = self.answers['answer'][self.questions['class'][self.identify(question)]]
        #print(f"Answer: {ans}")
        return ans
    
    def direct_answer(self, question):
        if self.answers is None:
            warnings.warn("Answers are not available")
            return None
        ans = self.answers['answer'][self.identify_direct_answer(question)]
        #print(f"Answer: {ans}")
        return ans

    # -------------------- Added --------------------

    def get_same_question_different_answer_pairs(self, save_path=None):
        cm = self.db @ self.db.T
        am = np.argmax(cm, axis=1)
        dict_question2classes = {}
        print("Same question with different classes:")
        for i in range(am.shape[0]):
            if am[i] != i:
                if self.questions["question"][i] not in dict_question2classes:
                    dict_question2classes[self.questions["question"][i]] = []
                if str(self.questions["class"][i]) not in dict_question2classes[self.questions["question"][i]]:
                    dict_question2classes[self.questions["question"][i]].append(str(self.questions["class"][i]))
                if str(self.questions["class"][am[i]]) not in dict_question2classes[self.questions["question"][i]]:
                    dict_question2classes[self.questions["question"][i]].append(str(self.questions["class"][am[i]]))
                print("\nQ:", self.questions["question"][i], "It's idx=", i, ", true_class=", self.questions["class"][i])
                print("Q_argmax:", self.questions["question"][am[i]], "It's idx= ", am[i], ", true_class=",self.questions["class"][am[i]])
        
        if save_path is not None:
            json_data = json.dumps(dict_question2classes)
            with open(save_path, "w") as outfile:
                outfile.write(json_data)
        
        return dict_question2classes

    def get_most_confused_questions(self, cos_sim_threshold: float, save_path=None):
        # With mean-match
        cm = self.db @ self.mean_db.T
        print(cm.shape) # (questions, classes)
        masked_with_thresh = cm > cos_sim_threshold

        # For each question: find the number of mean vectors that have larger cosine
        # similarity with given question than threshold.
        n_of_similar_questions_to_question = masked_with_thresh.sum(axis=1)
        print(n_of_similar_questions_to_question.shape)

        # Find indices of questions that are close to being missclassified.
        # Questions that have multiple mean vectors that yield cosine > threshold.
        idxs_close_miss = np.where(n_of_similar_questions_to_question > 1)

        for i in idxs_close_miss:
            print(self.questions["question"][i])

    def get_most_misclassified_class_pairs(self, save_path=None, n_of_common_misses=3):
        cm = self.db @ self.mean_db.T
        am = np.argmax(cm, axis=1)

        class_to_class = {}

        preds = am
        gts = self.questions["class"].to_numpy(dtype=int)
        hits = preds == gts
        for i, b in enumerate(hits):
            if not b:
                # print(f"{self.questions['question'][i]} : Class {am[i]}")
                if (self.questions['class'][i], am[i]) in class_to_class:
                    class_to_class[(self.questions['class'][i], am[i])] += 1
                elif (am[i], self.questions['class'][i]) in class_to_class:
                    class_to_class[(am[i], self.questions['class'][i])] += 1
                else:
                    class_to_class[(self.questions['class'][i], am[i])] = 1
                

        for pair in class_to_class:
            if class_to_class[pair] >= n_of_common_misses:
                print(pair, class_to_class[pair]) # class pair, number of common misses

        new_cl2cl = [str(k[0])+":"+str(k[1]) for k, v in class_to_class.items() if v >= n_of_common_misses]
        print(new_cl2cl)

        if save_path:
            with open(save_path, "w") as outfile:
                json.dump(new_cl2cl, outfile)

    def mean_match_test_disjunctive(self, leave_one_out_also_tfidf=False):
        # does not include the tested question into "training data"
        copy_probs = self.word_probs
        n_got_right = 0
        n_got_second_right = 0
        n_got_third_right = 0
        for i, cls in enumerate(self.questions["class"].unique()):
            # i == cls
            for index, row in self.questions[self.questions["class"] == cls].iterrows():
                tmp_mean_db = self.mean_db
                tmp_mean_vec = np.zeros(tmp_mean_db.shape[1])
                c = 0
                if self.tfidf_weighting and leave_one_out_also_tfidf:
                    classifier = TFIDF_Classifier(self.path_to_q)
                    test_data = classifier.structure_data(test_data_percent=1, sents_idxs_to_leaveout=[index]) 
                    tfidf_matrix, feat_names = classifier.get_TFIDF_matrix()
                    self.word_probs = get_TFIDF_threshold_probabilities(tfidf_matrix, feat_names)

                for index_m, row_m in self.questions[self.questions["class"] == cls].iterrows():
                    if index == index_m:
                        continue
                    tmp_mean_vec += self.sentence_embedding(row_m["question"])
                    c += 1
                tmp_mean_vec /= c

                tmp_mean_db[i] = tmp_mean_vec
                cm = self.db[index] @ tmp_mean_db.T
                sorted_indexes = np.argsort(cm)
                pred = sorted_indexes[-1]

                if pred == cls:
                    n_got_right += 1
                elif sorted_indexes[-2] == cls:
                    n_got_second_right += 1
                elif sorted_indexes[-3] == cls:
                    n_got_third_right += 1
        
        self.word_probs = copy_probs
        n_of_questions = len(self.questions.index)
        return round(n_got_right / n_of_questions, 3), round(n_got_second_right / n_of_questions, 3),  round(n_got_third_right / n_of_questions, 3)

    def cross_match_test_tfidf_disj(self):
        # Computes cosine similarities of all question pairs
        # A question is succesfully matched, if its second highest similarity is with a question of the same class
        # Computes accuracy as the ratio of succesfull matches
        copy_probs = self.word_probs
        cls_ids = self.questions["class"].to_numpy(dtype=int)

        hits = np.zeros(len(self.questions))
        hits2 = np.zeros(len(self.questions))
        mask = np.full(len(self.questions), False)
        diff_prediction_right = 0

        for i, row in enumerate(self.questions.iterrows()):
            c = TFIDF_Classifier(self.path_to_q)
            c.structure_data(test_data_percent=1, sents_idxs_to_leaveout=[i]) 
            tfidf_matrix, feat_names = c.get_TFIDF_matrix()
            self.word_probs = get_TFIDF_threshold_probabilities(tfidf_matrix, feat_names)
            self.db = np.array([self.sentence_embedding(q) for q in self.questions["question"]])

            cm = self.db[i] @ self.db.T
            am = np.argsort(cm)[-2]

            mask[i] = True
            hits += (cls_ids == cls_ids[am]) * mask

            # Second nearest neighbour
            am2 = np.argsort(cm)[-3]
            hits2 += (cls_ids == cls_ids[am2]) * (hits == False) * mask
            mask[i] = False


            "didn't really help"
            # def compare_ham_distance(sent, cls1, cls2):
            #     c_in1 = 0
            #     c_in2 = 0
            #     for w in LMTZR.clean_corpus(sent):
            #         if w in c.structured_list[cls1]:
            #             c_in1 += 1
            #         if w in c.structured_list[cls2]:
            #             c_in2 += 1
            #     if c_in1 >= c_in2:
            #         return cls1
            #     else:
            #         return cls2

            # if cm[am] < 0.7:
            #     if compare_ham_distance(self.questions["question"][i], cls_ids[am], cls_ids[am2]) == cls_ids[i]:
            #         diff_prediction_right += 1
            # elif cls_ids[am] == cls_ids[i]:
            #     diff_prediction_right += 1


        self.word_probs = copy_probs
        return round(hits.mean(), 3), round(hits2.mean(), 3)
    # , diff_prediction_right/len(self.questions)

