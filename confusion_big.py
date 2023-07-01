
import fasttext
import argparse
import pickle

import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class FAQ:
    def __init__(self, model, questions, answers_path=None, alpha=None, svd=False, corpus_size=4.1e9):
        self.model = model
        self.answers = None
        self.sentence_embedding = self.default_sentence_embedding
        self.word_probs = None
        self.alpha = alpha
        self.svd = svd
        self.questions = questions
        # if questions_path.split(".")[1] == "xlsx":
        #     self.questions = pd.read_excel(questions_path)
        # elif questions_path.split(".")[1] == "json":
        #     self.questions = pd.read_json(questions_path)
        # else:
        #     raise "Unsupported data file"

        self.questions = pd.DataFrame(self.questions)

        if answers_path and questions_path.split(".")[1] == "xlsx":
            self.answers = pd.read_excel(answers_path)
        elif answers_path and questions_path.split(".")[1] == "json":
            self.answers = pd.read_json(answers_path)
        elif answers_path:
            raise "Unsupported data file"

        if alpha is not None:
            self.sentence_embedding = self.weighted_sentence_embedding
            words, freqs = model.get_words(include_freq=True)
            probs = list(map(lambda x: float(x) / corpus_size, freqs))
            #print(words[:5], probs[:5])
            self.word_probs = dict(zip(words, probs))

        self.db = np.array([self.sentence_embedding(q) for q in self.questions["question"]])
        self.mean_db = np.zeros([self.questions["class"].nunique(), self.db.shape[1]])
        for i, cls in enumerate(self.questions["class"].unique()):
            imin = self.questions[self.questions["class"] == cls].index.min()
            imax = self.questions[self.questions["class"] == cls].index.max()
            self.mean_db[i, :] = self.db[imin:imax+1, :].mean(axis=0)
        if self.answers is not None:
            self.ans_db = np.array([self.sentence_embedding(a) for a in self.answers['answer']])

        def singular_decouple(mat):
            u, s, vh = np.linalg.svd(mat.T)
            u = u[:, :1]
            mat -= (u @ u.T @ mat.T).T
            return mat

        if alpha is not None and self.svd:
            self.db = singular_decouple(self.db)
            self.mean_db = singular_decouple(self.mean_db)
            if self.answers is not None:
                self.ans_db = singular_decouple(self.ans_db)

    def default_sentence_embedding(self, sentence):
        embedding = self.model.get_sentence_vector(sentence.lower().replace('\n', ' '))
        return embedding/np.linalg.norm(embedding)

    def mean_sentence_embedding(self, sentence):
        # Same as default, but computed manually
        words = sentence.lower().replace('\n', ' ').split()
        wes = np.array([self.model.get_word_vector(w) for w in words])
        wes /= np.linalg.norm(wes, axis=1)[:, np.newaxis]
        se = np.mean(wes, axis=0)
        return se/np.linalg.norm(se)

    def weighted_sentence_embedding(self, sentence):
        def word_probability(word):
            if word in self.word_probs.keys():
                return self.word_probs[word]
            return 0.0

        words = sentence.lower().replace('\n', ' ').split()
        wes = np.array([self.model.get_word_vector(w) for w in words])
        probs = np.array([word_probability(w) for w in words])[:, np.newaxis]
        wes /= np.linalg.norm(wes, axis=1)[:, np.newaxis] + 1e-9
        wes *= self.alpha / (self.alpha + probs)
        se = np.mean(wes, axis=0)
        return se/np.linalg.norm(se)

    def total_confusion(self):
        cm = self.db @ self.db.T
        am = np.argmax(cm, axis=1)
        for i in range(am.shape[0]):
            if am[i] != i:
                print("Ambiguous match:")
                print(self.questions["question"][i], i, self.questions["class"][i])
                print(self.questions["question"][am[i]], am[i], self.questions["class"][am[i]])
                print()

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

    def mean_match_test(self, verb=False, show_cm=False):
        cm = self.db @ self.mean_db.T
        am = np.argmax(cm, axis=1)
        preds = am
        gts = self.questions["class"].to_numpy(dtype=int)
        hits = preds == gts

        acc = hits.mean()
        print(f"Mean match accuracy: {acc}")

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
            plt.pause(0.1)
            return acc, fig
        return acc, None
    
    def cross_match_test(self, verb=False, show_cm=False):
        cm = self.db @ self.db.T
        am = np.argsort(cm, axis=1)[:, -2]
        cls_ids = self.questions["class"].to_numpy(dtype=int)
        hits = cls_ids == cls_ids[am]

        acc = hits.mean()
        print(f"Question cross-match accuracy: {acc}")

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
            plt.pause(0.1)
            return acc, fig
        return acc, None

    def ans_test(self, verb=False, show_cm=False):
        if self.answers is None:
            warnings.warn("Answers are not available")
            return None
        cm = self.db @ self.ans_db.T
        am = np.argmax(cm, axis=1)
        preds = self.answers["class"].to_numpy(dtype=int)[am]
        gts = self.questions["class"].to_numpy(dtype=int)
        hits = preds == gts

        acc = hits.mean()
        print(f"Answer match accuracy: {acc}")

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
            plt.pause(0.1)
            return acc, fig
        return acc, None
    
    
if __name__=='__main__':
    
    with open('../upv_search/expanded_data_all2.pkl', 'rb') as f:
        data = pickle.load(f)
        
    # questions = []
    # for idx, ans in enumerate(data):
    #     for q in data[ans]['questions']:
    #         if len(q) > 3:
    #             questions.append({'question': q, 'class': idx})
                
    questions = {"question":[], "class":[]}
    for idx, ans in enumerate(data):
        for q in data[ans]['questions']:
            if len(q) > 3:
                # questions.append({'question': q, 'class': idx})
                questions['question'].append(q)
                questions['class'].append(idx)
    
    model = fasttext.load_model('../upv_search/cbow_300_ns10_800k_ccfin_ep10.bin')
    faq = FAQ(model, questions)
    faq.total_confusion()
