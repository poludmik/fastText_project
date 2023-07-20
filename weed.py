from faq50_adapted import *
import math
from scipy import spatial


def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def cosine_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    return 1 - spatial.distance.cosine(vec1, vec2)


class WEED(FAQ):
    """
    Word Embeddings based Edit Distance
    """
    def __init__(self, model, questions_path, 
                 answers_path=None, 
                 probs=None, 
                 alpha=0.0001, 
                 compressed=False, 
                 rm_stop_words=False, 
                 lemm=False, 
                 tfidf_weighting=False):
        super().__init__(model, questions_path, answers_path, probs, alpha, compressed, rm_stop_words, lemm, tfidf_weighting)
        self.word_embs_db = [np.array([self.get_w_vec(w) for w in LMTZR.clean_corpus(q, rm_stop_words=rm_stop_words, lemm=lemm)]) for q in self.questions["question"]]
        
        def word_probability(word):
            if word in self.word_probs.keys():
                return self.word_probs[word]
            return 0.0
        
        self.word_probs_db = [np.array([word_probability(w) for w in LMTZR.clean_corpus(q, rm_stop_words=rm_stop_words, lemm=lemm)])[:, np.newaxis] for q in self.questions["question"]]

    def nearest_question_test_weed(self, sw=False, lm=False):
        # s1 = "jaké informace najdu ve Věstníku"
        # s2 = "jaké jablko ve najdu Věstníku"
        # print(s1)
        # print(s2, "\n")
        cm = np.zeros((len(self.word_embs_db), len(self.word_embs_db))) # 2000 x 2000

        for i, sent_embs in enumerate(self.word_embs_db):
            for j, sent_ref_embs in enumerate(self.word_embs_db):
                # cm[i, j] = self.ed_between_two_sentences(sent_embs, sent_ref_embs)
                cm[i, j] = self.distance(sent_embs, sent_ref_embs, self.word_probs_db[i], self.word_probs_db[j])

        cls_ids = self.questions["class"].to_numpy(dtype=int)
        am = np.argsort(cm, axis=1)[:, -2]
        # print(cls_ids[am])
        hits = cls_ids == cls_ids[am]
        acc = hits.mean()
        return acc


    def distance(self, sent_embs1, sent_embs2, s1_probs=1, s2_probs=1):
        multiplied = (s1_probs*sent_embs1) @ (s2_probs*sent_embs2).T
        max_for_each_w1 = np.max(multiplied, axis=1)
        return np.mean(max_for_each_w1)
    
        # max_for_each_w1 = np.max(sent_embs1 @ sent_embs2.T, axis=1) # Does 0.218 accuaracy
        # #  / np.linalg.norm(max_for_each_w1)
        # return np.sum(max_for_each_w1) / sent_embs1.shape[0]




    # def ed_between_two_sentences(self, sent1_embs, sent2_embs):
    #     dp_matrix = np.zeros((len(sent1_embs) + 1, len(sent2_embs) + 1))

    #     # fill first column and first row with increasing numbers (+1 insertions)
    #     dp_matrix[:, 0] = np.arange(len(sent1_embs) + 1)
    #     dp_matrix[0, :] = np.arange(len(sent2_embs) + 1) 

    #     # dynamic programming matrix filling
    #     for i_, w_i in enumerate(sent1_embs):
    #         for j_, w_j in enumerate(sent2_embs):
    #             i = i_ + 1
    #             j = j_ + 1
    #             # dp_matrix[i, j] = dp_matrix[i-1, j-1] + WEED.substitute(w_i, w_j)
    #             dp_matrix[i, j] = min(dp_matrix[i, j-1] + WEED.ins_del(sent1_embs, w_i, w_j), 
    #                                   dp_matrix[i-1, j] + WEED.ins_del(sent2_embs, w_j, w_i),
    #                                   dp_matrix[i-1, j-1] + WEED.substitute(w_i, w_j)
    #                                   )
    #     return dp_matrix[-1, -1]


    @staticmethod
    def ins_del(S_A, w_A_i, w_B_j, lamb=0.5, gamma=0.2):
        if S_A.shape[0] == 1:
            return 1
        return 1 - (lamb*max([WEED.sim(wA_k, w_B_j) for wA_k in S_A if np.any(wA_k != w_A_i)]) + gamma)


    @staticmethod
    def substitute(w_A_i, w_B_j):
        return 2 - 2 * WEED.sim(w_A_i, w_B_j)


    @staticmethod
    def sim(word1_emb, word2_emb, w=10, b=0.1):
        # if np.all(word1_emb == word2_emb): # could slow down significantly
        #     return 1.0
        return sigmoid(w * cosine_sim(word1_emb, word2_emb) + b)
        
        
