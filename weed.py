import copy
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
                 sigma=0.6,
                 compressed=False, 
                 rm_stop_words=False, 
                 lemm=False, 
                 tfidf_weighting=False,
                 slBert=False):
        super().__init__(model, questions_path, answers_path, probs, alpha, compressed, rm_stop_words, lemm, tfidf_weighting, slBert)
        
        self.sigma = sigma

        def word_probability(word):
            if word in self.word_probs.keys():
                return self.word_probs[word]
            # return 0
            return min(self.word_probs.items(), key=lambda x: x[1])[1]

        questions = [q for q in self.questions["question"]]
        self.tokenized_sents = [LMTZR.clean_corpus(q, rm_stop_words=rm_stop_words, lemm=lemm) for q in questions]
        if slBert:
            self.word_embs_db = [np.array(model.get_mean_sentence_embedding(q, mean=False)) for q in questions]
            self.word_probs_db = [np.array([word_probability(w) for w in self.model.tokenizer.tokenize(q)])[:, np.newaxis] for q in questions]
        else:
            self.word_embs_db = [np.array([self.get_w_vec(w)/(np.linalg.norm(self.get_w_vec(w))+1e-9) for w in LMTZR.clean_corpus(q, rm_stop_words=rm_stop_words, lemm=lemm)]) for q in questions]
            self.word_probs_db = [np.array([word_probability(w) for w in LMTZR.clean_corpus(q, rm_stop_words=rm_stop_words, lemm=lemm)])[:, np.newaxis] for q in questions]
        # print("Average number of words in a tokenized sentence:", round(np.mean(np.array([len(x) for x in self.word_probs_db])), 3))


    def nearest_question_test_weed(self):
        cm = np.zeros((len(self.word_embs_db), len(self.word_embs_db))) # n_of_questions x n_of_questions

        for i, sent_embs in enumerate(self.word_embs_db):
            for j, sent_ref_embs in enumerate(self.word_embs_db):
                # cm[i, j] = self.ed_between_two_sentences(sent_embs, sent_ref_embs) # WED algorithm, slow
                ss = self.semantic_similarity(sent_embs, sent_ref_embs, self.word_probs_db[i], self.word_probs_db[j])
                wos = self.word_order_similarity(self.tokenized_sents[i], self.tokenized_sents[j])
                cm[i, j] = self.sigma * ss + (1 - self.sigma) * wos

        np.fill_diagonal(cm, -np.inf)

        am = np.argsort(cm, axis=1)[:, -1]
        cls_ids = self.questions["class"].to_numpy(dtype=int)
        # print(cls_ids[am])
        hits = cls_ids == cls_ids[am]
        acc = hits.mean()
        return acc

    def semantic_similarity(self, sent_embs1, sent_embs2, s1_probs=1, s2_probs=1):
        
        if self.tfidf_weighting:
            multiplied = (s1_probs/(self.alpha+s1_probs) * sent_embs1) @ (s2_probs/(self.alpha+s2_probs) * sent_embs2).T
        elif self.slBert:
            multiplied = (sent_embs1) @ (sent_embs2).T
        else:
            multiplied = (1/(self.alpha+s1_probs) * sent_embs1) @ (1/(self.alpha+s2_probs) * sent_embs2).T

        max_for_each_w1 = np.max(multiplied, axis=1)
        ss = np.mean(max_for_each_w1)

        if self.tfidf_weighting:
            multiplied_2 = (s2_probs/(self.alpha+s2_probs) * sent_embs2) @ (s1_probs/(self.alpha+s1_probs) * sent_embs1).T
        elif self.slBert:
            multiplied_2 = (sent_embs2) @ (sent_embs1).T
        else:
            multiplied_2 = (1/(self.alpha+s2_probs) * sent_embs2) @ (1/(self.alpha+s1_probs) * sent_embs1).T

        max_for_each_w1_2 = np.max(multiplied_2, axis=1)
        ss_2 = np.mean(max_for_each_w1_2)

        return (ss + ss_2) / 2
    
    def word_order_similarity(self, sent1_t, sent2_t):
        union = list(set(sent1_t + sent2_t))
        V1 = np.zeros(len(union))
        V2 = np.zeros(len(union))

        for i, w in enumerate(union):
            if w in sent1_t:
                V1[i] = sent1_t.index(w)
            if w in sent2_t:
                V2[i] = sent2_t.index(w)
        return 1 - (np.linalg.norm(V1 - V2) / (np.linalg.norm(V1 + V2) + 1e-9))





        


    # ------------- WED: slow --------------

    def ed_between_two_sentences(self, sent1_embs, sent2_embs):
        dp_matrix = np.zeros((len(sent1_embs) + 1, len(sent2_embs) + 1))

        # fill first column and first row with increasing numbers (+1 insertions)
        dp_matrix[:, 0] = np.arange(len(sent1_embs) + 1)
        dp_matrix[0, :] = np.arange(len(sent2_embs) + 1) 

        # dynamic programming matrix filling
        for i_, w_i in enumerate(sent1_embs):
            for j_, w_j in enumerate(sent2_embs):
                i = i_ + 1
                j = j_ + 1
                # dp_matrix[i, j] = dp_matrix[i-1, j-1] + WEED.substitute(w_i, w_j)
                dp_matrix[i, j] = min(dp_matrix[i, j-1] + WEED.ins_del(sent1_embs, w_i, w_j), 
                                      dp_matrix[i-1, j] + WEED.ins_del(sent2_embs, w_j, w_i),
                                      dp_matrix[i-1, j-1] + WEED.substitute(w_i, w_j)
                                      )
        return dp_matrix[-1, -1]

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
        # if np.all(word1_emb == word2_emb): # could slow down
        #     return 1.0
        return sigmoid(w * cosine_sim(word1_emb, word2_emb) + b)
        
        
