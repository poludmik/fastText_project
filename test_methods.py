import fasttext
from faq50_adapted import FAQ, extract_word_probs
from topic_word_probs import *
from tfidf_classifier import TFIDF_Classifier
from cs_lemmatizer import *


class Tester:

    def __init__(self) -> None:
        self.TFIDF_leave_one_out_acc = {}
        self.TFIDF_n_of_words = None

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



if __name__ == "__main__":
    # Given:
    model_path = "models/cc.cs.300.bin"
    path_to_q = "780_upv_questions/Q78_questions.xlsx"
    path_to_a = "780_upv_questions/Q78_answers_no_tags.xlsx"
    # path_to_save_results = "test_results/results.xlsx"


    t = Tester()
    t.TFIDF_classifier_test(path_to_q)



