import fasttext
import numpy as np
import argparse
import os

from tqdm import tqdm
from dataclasses import dataclass
from faq50_adapted import FAQ, extract_word_probs
from topic_word_probs import *
from tfidf_classifier import TFIDF_Classifier
from cs_lemmatizer import *
from weed import *



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


    def fast_text_tests(self, model_path, path_to_q, path_to_a, path_to_save,
                        mean=False, 
                        weighted=False,
                        weighted_qa=False,
                        weighted_by_tfidf=False,
                        weed=False
                        ):
        
        if os.path.exists(path_to_save):
            print("Will add new columns or modify old ones in given <path_to_save>.xlsx file.")
            self.ft_results_df = pd.read_excel(path_to_save)

        stop_words_and_lemm = [(False, False), (True, False), (False, True), (True, True)]

        model = fasttext.load_model(model_path)

        if mean:
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
                self.ft_results_df.to_excel(path_to_save, index=False)

        if weighted:
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
                res = [str(best_a_cross.acc[0])+', '+str(best_a_cross.acc[1])+' : '+str(round(best_a_cross.alpha, 2)), 
                       str(best_a_mean.acc[0])+', '+str(best_a_mean.acc[1])+' : '+str(round(best_a_mean.alpha, 2)),
                       str(best_a_mean_disj.acc[0])+', '+str(best_a_mean_disj.acc[1])+', '+str(best_a_mean_disj.acc[2])+' : '+str(round(best_a_mean_disj.alpha, 2))]

                self.ft_results_df["Weighted_sent_embd: sw="+str(sw*1)+", lm="+str(lemm*1)] = res
                self.ft_results_df.to_excel(path_to_save, index=False)

        if weighted_qa:
            print("\nWeighted sentence embedding with data from q/a:")
            q_and_a = [(path_to_q, None), (None, path_to_a), (path_to_q, path_to_a)]
            for q, ans in q_and_a:
                probs, _ = count_word_probs_in_corpuses(path_to_questions=q, path_to_answers=ans)
                best_a_cross = AlphaAcc(0, [0, 0])
                best_a_mean = AlphaAcc(0, [0, 0])
                best_a_mean_disj = AlphaAcc(0, [0, 0, 0])

                # 30 different alphas to find the approximate best
                for alpha in tqdm(np.arange(0.01, 0.6, 0.02)):
                    faq = FAQ(model, path_to_q, path_to_a, probs=probs, alpha=alpha, rm_stop_words=True, lemm=True)

                    cross_acc, cross_acc_sec = faq.cross_match_test()
                    if cross_acc > best_a_cross.acc[0]:
                        best_a_cross = AlphaAcc(alpha, [cross_acc, cross_acc_sec])

                    mean_acc, mean_acc_sec = faq.mean_match_test()
                    if mean_acc > best_a_mean.acc[0]:
                        best_a_mean = AlphaAcc(alpha, [mean_acc, mean_acc_sec])

                    mean_disj_acc, s, t = faq.mean_match_test_disjunctive()
                    if mean_disj_acc > best_a_mean_disj.acc[0]:
                        best_a_mean_disj = AlphaAcc(alpha, [mean_disj_acc, s, t])

                a, b = 0, 0
                if q:
                    a = 1
                if ans:
                    b = 1
                print(f"From: que={a}, ans={b}, mean_disj={best_a_mean_disj.acc[0]}, best_a={best_a_mean_disj.alpha}")
                res = [str(best_a_cross.acc[0])+', '+str(best_a_cross.acc[1])+' : '+str(round(best_a_cross.alpha, 2)), 
                       str(best_a_mean.acc[0])+', '+str(best_a_mean.acc[1])+' : '+str(round(best_a_mean.alpha, 2)),
                       str(best_a_mean_disj.acc[0])+', '+str(best_a_mean_disj.acc[1])+', '+str(best_a_mean_disj.acc[2])+' : '+str(round(best_a_mean_disj.alpha, 2))]

                self.ft_results_df["Weighted_data_from q="+str(a)+", a="+str(b)+", sw=1, lm=1"] = res
                self.ft_results_df.to_excel(path_to_save, index=False)

        if weighted_by_tfidf:
            print("Weighted by TF-IDF:")
            
            best_a_cross = AlphaAcc(0, [0, 0])
            best_a_mean = AlphaAcc(0, [0, 0])

            c = TFIDF_Classifier(path_to_q)
            test_data = c.structure_data(test_data_percent=1) 
            tfidf_matrix, feat_names = c.get_TFIDF_matrix()
            probs = get_TFIDF_threshold_probabilities(tfidf_matrix, feat_names)

            # 30 different alphas to find the approximate best
            for alpha in tqdm(np.arange(0.01, 0.5, 0.02)):
                faq = FAQ(model, path_to_q, path_to_a, probs=probs, alpha=alpha, rm_stop_words=True, lemm=True,
                                            tfidf_weighting=True)
                
                cross_acc, cross_acc_sec = faq.cross_match_test()
                if cross_acc > best_a_cross.acc[0]:
                    best_a_cross = AlphaAcc(alpha, [cross_acc, cross_acc_sec])

                mean_disj_acc, s, t = faq.mean_match_test_disjunctive()
                if mean_disj_acc > best_a_mean.acc[0]:
                    best_a_mean = AlphaAcc(alpha, [mean_disj_acc, s, t])

            print(f"    Best alphas on non-disjunctive: cross:{best_a_cross.alpha}, mean:{best_a_mean.alpha}")

            res = [str(best_a_cross.acc[0])+', '+str(best_a_cross.acc[1])+' : '+str(round(best_a_cross.alpha, 2)), 
                       "Not tested",
                       str(best_a_mean.acc[0])+', '+str(best_a_mean.acc[1])+', '+str(best_a_mean.acc[2])+' : '+str(round(best_a_mean.alpha, 2))]

            self.ft_results_df["Weighted_by_tfidf (q included)"] = res


            print(f"Now testing disjunctive cross and mean with best alphas.\nIt takes 4-6 minutes on Q78 and ~30 minutes on Q76.")
            faq = FAQ(model, path_to_q, path_to_a, probs=probs, alpha=best_a_cross.alpha, rm_stop_words=True, lemm=True,
                                            tfidf_weighting=True)
            acc, acc_sec = faq.cross_match_test_tfidf_disj()
            faq = FAQ(model, path_to_q, path_to_a, probs=probs, alpha=best_a_mean.alpha, rm_stop_words=True, lemm=True,
                                            tfidf_weighting=True)
            f, s, t = faq.mean_match_test_disjunctive(leave_one_out_also_tfidf=True)

            res2 = [str(acc)+', '+str(acc_sec)+' : '+str(round(best_a_cross.alpha, 2)), 
                       "Not tested",
                       str(f)+', '+str(s)+', '+str(t)+' : '+str(round(best_a_mean.alpha, 2))]

            self.ft_results_df["Weighted_by_tfidf (disjunctive)"] = res2
            self.ft_results_df.to_excel(path_to_save, index=False)

        if weed:
            print("Word Embeddings based Edit Distance(similarity):")

            rm_sw = True
            lm = True

            best_a = AlphaAcc(0, [0])
            best_s = 1

            # probs from q/a
            print("probs from q/a:")
            probs, _ = count_word_probs_in_corpuses(path_to_questions=path_to_q)
            for a in tqdm(np.arange(0.0, 1, 0.1)): # 10 different alphas
                for s in np.arange(0.0, 1.0, 0.1):
                    weed = WEED(model, path_to_q, path_to_a, probs=probs, alpha=a, 
                                lemm=lm, rm_stop_words=rm_sw, sigma=s, 
                                tfidf_weighting=False)
                    acc = weed.nearest_question_test_weed()
                    if acc >= best_a.acc[0]:
                        best_a = AlphaAcc(a, [acc])
                        best_s = s
            print(f"best_acc={best_a.acc[0]}, best_a={best_a.alpha}, best_s={best_s}")
            res = [str(round(best_a.acc[0], 2)) + " : a=" + str(round(best_a.alpha, 2)) + ", s="+str(round(best_s, 2)), 
                   "",
                   ""]
            self.ft_results_df["WordEmbEditDist with p from q/a"] = res
            self.ft_results_df.to_excel(path_to_save, index=False)


            # probs from tf-idf:
            print("probs from tf-idf:")
            best_a = AlphaAcc(0, [0])
            best_s = 1
            c = TFIDF_Classifier(path_to_q, rm_sw, lm)
            test_data = c.structure_data(test_data_percent=1) 
            tfidf_matrix, feat_names = c.get_TFIDF_matrix()
            probs = get_TFIDF_threshold_probabilities(tfidf_matrix, feat_names)
            for a in tqdm(np.arange(0.05, 1.05, 0.1)): # 10 different alphas
                for s in np.arange(0.0, 1.0, 0.1):
                    weed = WEED(model, path_to_q, path_to_a, probs=probs, alpha=a, 
                                lemm=lm, rm_stop_words=rm_sw, sigma=s, 
                                tfidf_weighting=True)
                    acc = weed.nearest_question_test_weed()
                    if acc >= best_a.acc[0]:
                        best_a = AlphaAcc(a, [acc])
                        best_s = s
            print(f"best_acc={best_a.acc[0]}, best_a={best_a.alpha}, best_s={best_s}")
            res = [str(round(best_a.acc[0], 2)) + " : a=" + str(round(best_a.alpha, 2)) + ", s="+str(round(best_s, 2)), 
                   "",
                   ""]

            self.ft_results_df["WordEmbEditDist with p from tf-idf"] = res
            self.ft_results_df.to_excel(path_to_save, index=False)

        print(self.ft_results_df)
        self.ft_results_df.to_excel(path_to_save, index=False)



parser = argparse.ArgumentParser()
parser.add_argument("model_path", default=None, type=str, help="Path to .bin model file")
parser.add_argument("path_to_q", default=None, type=str, help="path to .xlsx with questions")
parser.add_argument("path_to_a", default=None, type=str, help="Path to .xlsx with answers")
parser.add_argument("save_path", default=None, type=str, help="Path to .xlsx with answers")

parser.add_argument("--mean", default=False, action=argparse.BooleanOptionalAction, help="Enable simple mean embedding test")
parser.add_argument("--weighted", default=False, action=argparse.BooleanOptionalAction, help="Enable weighted mean embedding test")
parser.add_argument("--weighted_qa", default=False, action=argparse.BooleanOptionalAction, help="Enable weighted test with corpus from q or/and a")
parser.add_argument("--weighted_tfidf", default=False, action=argparse.BooleanOptionalAction, help="Enable weighting by TF-IDF test")
parser.add_argument("--weed", default=False, action=argparse.BooleanOptionalAction, help="Enable word embedding based on edit distance tests")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    t = Tester()
    t.fast_text_tests(args.model_path, args.path_to_q, args.path_to_a, args.save_path,
                      mean=args.mean,
                      weighted=args.weighted,
                      weighted_qa=args.weighted_qa,
                      weighted_by_tfidf=args.weighted_tfidf,
                      weed=args.weed
                      )


    # ********************** Or manually without command line *********************:

    # # ----------------- Given small Q50 dataset:
    # model_path = "models/cc.cs.300.bin"
    # path_to_q = "upv_faq/Q50_questions.xlsx"
    # path_to_a = "upv_faq/Q50_answers.xlsx"
    # path_to_save_results = "test_results/results_on_small_dataset.xlsx"
    # t = Tester()
    # t.fast_text_tests(model_path, path_to_q, path_to_a, path_to_save_results,
    #                 #   mean=True,
    #                 #   weighted=True,
    #                 #   weighted_qa=True,
    #                 #   weighted_by_tfidf=True
    #                   )

    # # ----------------- Given Q78 dataset:
    # model_path = "models/cc.cs.300.bin"
    # path_to_q = "780_upv_questions/Q78_questions.xlsx"
    # path_to_a = "780_upv_questions/Q78_answers_no_tags.xlsx"
    # path_to_save_results = "test_results/fastText_tests_results.xlsx"
    # t = Tester()
    # t.fast_text_tests(model_path, path_to_q, path_to_a, path_to_save_results,
    #                   mean=True,
    #                   weighted=True,
    #                   weighted_qa=True,
    #                   weighted_by_tfidf=False)
    
