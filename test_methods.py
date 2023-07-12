import fasttext
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from faq50_adapted import FAQ, extract_word_probs
from topic_word_probs import *
from tfidf_classifier import TFIDF_Classifier
from cs_lemmatizer import *


@dataclass
class AlphaAcc:
    alpha: float
    acc: list[float]

    def sum_top_n_acc(self, n):
        return sum(self.acc[:n])


class Tester:

    def __init__(self) -> None:
        self.TFIDF_leave_one_out_acc = {}
        self.TFIDF_n_of_words = None

        self.results_columns = {'test_method': ['cross_match', 'mean_match', 'mean_match_disj']}
        self.ft_results_df = pd.DataFrame(self.results_columns)

    def TFIDF_classifier_test(self, path_to_questions_xslx):
        print("---- Started 4 TF-IDF leave_one_out_tests. ----")

        c = TFIDF_Classifier(path_to_questions_xslx)
        self.TFIDF_leave_one_out_acc["no_stop_w, no_lemm"] = c.leave_one_out_test(rm_stop_words=False, lemm=False)
        print("no_stop_w, no_lemm accuracy:", self.TFIDF_leave_one_out_acc["no_stop_w, no_lemm"])

        self.TFIDF_leave_one_out_acc["stop_w, no_lemm"] = c.leave_one_out_test(rm_stop_words=True, lemm=False)
        print("stop_w, no_lemm accuracy:", self.TFIDF_leave_one_out_acc["stop_w, no_lemm"])

        self.TFIDF_leave_one_out_acc["no_stop_w, lemm"] = c.leave_one_out_test(rm_stop_words=False, lemm=True)
        print("no_stop_w, lemm accuracy:", self.TFIDF_leave_one_out_acc["no_stop_w, lemm"])

        self.TFIDF_leave_one_out_acc["stop_w, lemm"] = c.leave_one_out_test(rm_stop_words=True, lemm=True)
        print("stop_w, lemm accuracy:", self.TFIDF_leave_one_out_acc["stop_w, lemm"])
        
        test_data = c.structure_data(test_data_percent=1) # without removal from train data
        c.get_TFIDF_matrix()
        self.TFIDF_n_of_words = c.TFIDF_matrix.shape[1]


    def fast_text_tests(self, model_path, path_to_q, path_to_a, path_to_save):
        stop_words_and_lemm = [(False, False), (True, False), (False, True), (True, True)]

        model = fasttext.load_model(model_path)

        print("\nMean sentence embedding:")
        for sw, lemm in stop_words_and_lemm:
            faq = FAQ(model, path_to_q, path_to_a, probs=None, alpha=None, rm_stop_words=sw, lemm=lemm)
            cross_acc, cross_acc_sec = faq.cross_match_test()
            mean_acc, mean_acc_sec = faq.mean_match_test()
            f, s, t = faq.mean_match_test_disjunctive()
            print(f"sw={sw*1}, lemm={lemm*1}, mean_disj={f}")
            res = [str(cross_acc)+', '+str(cross_acc_sec), 
                   str(mean_acc)+', '+str(mean_acc_sec),
                   str(f)+', '+str(s)+', '+str(t)]
            self.ft_results_df['Mean_sent_embd: sw='+str(sw*1)+", lm="+str(lemm*1)] = res

        print("\nWeighted sentence embedding:")
        probs, _ = count_word_probs_in_corpuses(path_to_questions=path_to_q, path_to_answers=path_to_a)
        for sw, lemm in stop_words_and_lemm:
            best_a_cross = AlphaAcc(0, [0, 0])
            best_a_mean = AlphaAcc(0, [0, 0])
            best_a_mean_disj = AlphaAcc(0, [0, 0, 0])
            
            # 30 different alphas to find the approximate best
            for alpha in tqdm(np.arange(0.01, 0.6, 0.02)):
                faq = FAQ(model, path_to_q, path_to_a, probs=probs, alpha=alpha, rm_stop_words=sw, lemm=lemm)
                
                cross_acc, cross_acc_sec = faq.cross_match_test()
                if cross_acc > best_a_cross.acc[0]:
                    best_a_cross = AlphaAcc(alpha, [cross_acc, cross_acc_sec])

                mean_acc, mean_acc_sec = faq.mean_match_test()
                if mean_acc > best_a_mean.acc[0]:
                    best_a_mean = AlphaAcc(alpha, [mean_acc, mean_acc_sec])

                mean_disj_acc, s, t = faq.mean_match_test_disjunctive()
                if mean_disj_acc > best_a_mean_disj.acc[0]:
                    best_a_mean_disj = AlphaAcc(alpha, [mean_disj_acc, s, t])

            print(f"sw={sw}, lemm={lemm}, mean_disj={best_a_mean_disj.acc[0]}, best_a={best_a_mean_disj.alpha}")
            res = [str(best_a_cross.acc[0])+', '+str(best_a_cross.acc[1])+' : '+str(best_a_cross.alpha), 
                   str(best_a_mean.acc[0])+', '+str(best_a_mean.acc[1])+' : '+str(best_a_mean.alpha),
                   str(best_a_mean_disj.acc[0])+', '+str(best_a_mean_disj.acc[1])+', '+str(best_a_mean_disj.acc[2])+' : '+str(best_a_mean_disj.alpha)]
            
            self.ft_results_df["Weighted_sent_embd: sw="+str(sw*1)+", lm="+str(lemm*1)+":alpha"] = res

        print(self.ft_results_df)
        self.ft_results_df.to_excel(path_to_save)




if __name__ == "__main__":
    # Given:
    model_path = "models/cc.cs.300.bin"
    path_to_q = "780_upv_questions/Q78_questions.xlsx"
    path_to_a = "780_upv_questions/Q78_answers_no_tags.xlsx"
    path_to_save_results = "test_results/all_fastText_tests_results.xlsx"


    t = Tester()
    # t.TFIDF_classifier_test(path_to_q)
    t.fast_text_tests(model_path, path_to_q, path_to_a, path_to_save_results)



