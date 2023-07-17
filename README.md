# Testing usage of FastText embeddings for FAQ answering

## Main testing script - test_methods.py

> Evaluates several approaches for the task based on the given questions-class pairs and .bin model and outputs results to .xlsx file.

### Usage:
> test_methods.py [-h] [--mean | --no-mean] [--weighted | --no-weighted] [--weighted_qa | --no-weighted_qa] [--weighted_tfidf | --no-weighted_tfidf] model_path path_to_q path_to_a save_path

### Usage example (to run only "weighted" test):
> test_methods.py models/cc.cs.300.bin upv_faq/Q50_questions.xlsx upv_faq/Q50_answers.xlsx test_results/desired_results_filename.xlsx --weighted

Or it could be easier to change desired paths and parameters in the actual main() through some IDE.

### Arguments:
* **model_path**: Path to \<model\>.bin
* **path_to_q**: Path to .xlsx with question-class pairs
* **path_to_a**: Path to .xlsx with class-answer pairs
* **save_path**: Where to save the \<results\>.xlsx
* **--mean**: Run simple mean embedding test
* **--weighted**: Run tests with weighted by frequency embeddings:
* **--weighted_qa**: Same tests as previous, but tests the importance of taking frequencies from qustions or/and answers.
* **--weighted_tfidf**: Run tests with TF-IDF values as weights for sentence embeddings.

## TODO:
* What does it do
* What type of sentence embeddings does it produce
* What tests are evaluated
* Explain files/folders
* Cross/Mean/Disjunctive (disjunctive for tf-idf weights)

