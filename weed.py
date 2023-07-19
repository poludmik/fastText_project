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

    def test(self):
        s1 = "jaké informace najdu ve Věstníku"
        s2 = "jaké jablko ve najdu Věstníku"
        print(s1)
        print(s2, "\n")
        return self.ed_between_two_sentences(s1, s2)

    def ed_between_two_sentences(self, sent1: str, sent2: str, rm_sw=False, lm=False):
        s1 = [self.get_w_vec(w) for w in LMTZR.clean_corpus(sent1, rm_stop_words=rm_sw, lemm=lm)]
        s2 = [self.get_w_vec(w) for w in LMTZR.clean_corpus(sent2, rm_stop_words=rm_sw, lemm=lm)]
        dp_matrix = np.zeros((len(s1) + 1, len(s2) + 1))

        # fill first column and first row with increasing numbers (+1 insertions)
        dp_matrix[:, 0] = np.arange(len(s1) + 1)
        dp_matrix[0, :] = np.arange(len(s2) + 1) 

        for i_, w_i in enumerate(s1):
            # print(LMTZR.clean_corpus(sent1, rm_stop_words=rm_sw, lemm=lm)[i_]) # word i from A
            for j_, w_j in enumerate(s2):
                # print("     "+LMTZR.clean_corpus(sent1, rm_stop_words=rm_sw, lemm=lm)[j_]) # word j from B
                i = i_ + 1
                j = j_ + 1
                dp_matrix[i, j] = min(dp_matrix[i, j-1] + WEED.ins_del(s1, w_i, w_j), 
                                      dp_matrix[i-1, j] + WEED.ins_del(s2, w_j, w_i),
                                      dp_matrix[i-1, j-1] + WEED.substitute(w_i, w_j)
                                      )
                # print(np.argmin(np.array([dp_matrix[i, j-1] + WEED.ins_del(s1, w_i, w_j), 
                #                       dp_matrix[i-1, j] + WEED.ins_del(s2, w_j, w_i),
                #                       dp_matrix[i-1, j-1] + WEED.substitute(w_i, w_j)])))
                
        print(np.around(dp_matrix, 3))

        return dp_matrix[-1, -1]


    @staticmethod
    def ins_del(S_A, w_A_i, w_B_j, lamb=0.5, gamma=0.2):
        return 1 - (lamb*max([WEED.sim(wA_k, w_B_j) for wA_k in S_A if np.any(wA_k != w_A_i)]) + gamma)


    @staticmethod
    def substitute(w_A_i, w_B_j):
        return 2 - 2 * WEED.sim(w_A_i, w_B_j)


    @staticmethod
    def sim(word1_emb, word2_emb, w=10, b=0.1):
        # if np.all(word1_emb == word2_emb): # could slow down significantly
        #     return 1.0
        return sigmoid(w * cosine_sim(word1_emb, word2_emb) + b)
        
        
