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
#from nltk.tokenize import word_tokenize


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


class FAQ_adapted:
    def __init__(
            self,
            model, 
            questions_path, 
            answers_path=None,
            probs=None, 
            alpha=1e-4, 
            compressed=False
        ):
        self.model = model
        self.answers = None
        self.sentence_embedding = self.default_sentence_embedding
        self.word_probs = probs
        self.alpha = alpha

        if compressed:
            self.get_w_vec = self.model.word_vec
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

    def mean_sentence_embedding(self, sentence):
        # Same as default, but computed manually
        words = sentence.lower().replace('\n', ' ').split()
        #words = word_tokenize(sentence)
        wes = np.array([self.get_w_vec(w) for w in words])
        wes /= np.linalg.norm(wes, axis=1)[:, np.newaxis] + 1e-9
        se = np.mean(wes, axis=0)
        return se/np.linalg.norm(se)

    def weighted_sentence_embedding(self, sentence):
        # Computes weighted sentence embedding acoording to: https://openreview.net/pdf?id=SyK00v5xx
        def word_probability(word):
            if word in self.word_probs.keys():
                return self.word_probs[word]
            return 0.0

        words = sentence.lower().replace('\n', ' ').split()
        #words = word_tokenize(sentence)
        wes = np.array([self.get_w_vec(w) for w in words])
        probs = np.array([word_probability(w) for w in words])[:, np.newaxis]
        wes /= np.linalg.norm(wes, axis=1)[:, np.newaxis] + 1e-9
        wes *= self.alpha / (self.alpha + probs)
        se = np.mean(wes, axis=0)
        return se/np.linalg.norm(se)

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
        # print(self.db.shape)
        print(self.mean_db.shape)
        cm = self.db @ self.mean_db.T
        # print(cm.shape)
        am = np.argmax(cm, axis=1)
        preds = am
        gts = self.questions["class"].to_numpy(dtype=int)
        hits = preds == gts

        acc = hits.mean()
        #print(f"Mean match accuracy: {acc}")

        if not hits.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(hits):
                if not b:
                    print(f"{self.questions['question'][i]} : Class {am[i]}")

        if show_cm:
            cm = confusion_matrix(gts, preds)
            fig = plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=True)
            plt.title("Mean matching confusion matrix")
            plt.xlabel("Prediction")
            plt.ylabel("True class")
            plt.draw()
            plt.pause(show_time)
            return acc, fig
        return acc, None
    
    def cross_match_test(self, verb=False, show_cm=False, show_time=2.0):
        # Computes cosine similarities of all question pairs
        # A question is succesfully matched, if its second highest similarity is with a question of the same class
        # Computes accuracy as the ratio of succesfull matches
        cm = self.db @ self.db.T
        am = np.argsort(cm, axis=1)[:, -2]
        cls_ids = self.questions["class"].to_numpy(dtype=int)
        hits = cls_ids == cls_ids[am]

        acc = hits.mean()
        #print(f"Question cross-match accuracy: {acc}")

        if not hits.all() and verb:
            print("\nIncorrect matches:")
            for i, b in enumerate(hits):
                if not b:
                    print(f"{self.questions['question'][i]} : {self.questions['question'][am[i]]}")

        if show_cm:
            cm = confusion_matrix(cls_ids, cls_ids[am])
            fig = plt.figure(figsize=(10, 7))
            sn.heatmap(cm, annot=True)
            plt.title("Question cross-matching confusion matrix")
            plt.xlabel("Prediction")
            plt.ylabel("True class")
            plt.draw()
            plt.pause(show_time)
            return acc, fig
        return acc, None

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
        return matched_q

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

        for i in range(am.shape[0]):
            if am[i] != i:
                if self.questions["question"][i] not in dict_question2classes:
                    dict_question2classes[self.questions["question"][i]] = []
                if str(self.questions["class"][i]) not in dict_question2classes[self.questions["question"][i]]:
                    dict_question2classes[self.questions["question"][i]].append(str(self.questions["class"][i]))
                if str(self.questions["class"][am[i]]) not in dict_question2classes[self.questions["question"][i]]:
                    dict_question2classes[self.questions["question"][i]].append(str(self.questions["class"][am[i]]))
                # print("Same question with different classes:")
                # print("Q:", self.questions["question"][i], "It's idx=", i, ", true_class=", self.questions["class"][i])
                # print("Q_argmax:", self.questions["question"][am[i]], "It's idx= ", am[i], ", true_class=",self.questions["class"][am[i]])
        
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

        




