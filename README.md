# Testing usage of FastText embeddings for answering FAQ

## Main testing script - test_methods.py

Evaluates several approaches for the task based on the given questions-class pairs and a .bin model and outputs results to .xlsx file. Results achieved so far are in the `test_results` folder.

### Usage:
```
test_methods.py [-h] [--mean | --no-mean] [--weighted | --no-weighted] [--weighted_qa | --no-weighted_qa] [--weighted_tfidf | --no-weighted_tfidf] model_path path_to_q path_to_a save_path
```

### Usage example (to run only the "weighted" test):

```console
~$ test_methods.py models/cc.cs.300.bin upv_faq/Q50_questions.xlsx upv_faq/Q50_answers.xlsx test_results/desired_results_filename.xlsx --weighted
```

Or it could be easier to change desired paths and parameters in the actual main() through some IDE.

### Arguments:
* **model_path**: Path to \<model\>.bin
* **path_to_q**: Path to .xlsx with question-class pairs
* **path_to_a**: Path to .xlsx with class-answer pairs
* **save_path**: Where to save the \<results\>.xlsx
* **--mean**: Run a simple mean embedding test
* **--weighted**: Run tests with weighted by frequency embeddings:
* **--weighted_qa**: The same tests as previous but tests the importance of taking frequencies from questions or/and answers.
* **--weighted_tfidf**: Run tests with TF-IDF values as weights for sentence embeddings.

## What is the whole repository about
The main task to be solved is to create a FAQ bot that responds to some questions given predefined set of answers. User would provide a query, program will take it and will try to figure out the subject of the given query to answer with an appropriate answer.
Semantic text similarity methods are used to compare how much the query and some frequently asked questions differ. To represent each word, we use `fasttext` embeddings.

To find the FAQuestion that corresponds to the query the best, we create sentence embeddings. In this repository we try different approaches to create sentence embeddings from `fasttext` word embeddings. Also, there are two ways to find the best match that we test: cross_test and mean_test.

## Sentence embeddings from word embeddings
Given a sentence and word embeddings for each word, create an embedding that describes the whole sentence.
* **Mean embedding.** Take an average between word vectors.
* **Weighted by frequency.** Count the frequencies(calculate prior probabilities) of each word in the questions or answers corpuses and use them to weigh each word vector (weighted average). **α** parameter is introduced to tune the importance of probabilities.
* **Weighted by TF-IDF values.** Extract TF-IDF matrix from questions/answers corpuses and use word-class values V to weight the word embeddings by $V_w / (\alpha + V_w)$. So far, the value we use for a word is a maximal value that this word has across all classes. *If a word is important in some class, it should have a larger value*.

## Finding the best match
To find the according class for a given sentence embedding we have two methods.
* **Cross match.** For each training question in the FAQ database find it's sentence embedding. Then, find the nearest neighbor to a query sentence using cosine similarity.
* **Mean match.** For each class, create an average embedding and find the nearest neighbor from these *cluster centers*.

## Notes to `test_methods` testing parameters
1. A weighted test is called *disjunctive(disj)* if the query question is not used in average sentence vectors (separated train/test data). The probabilities are still calculated, including the query question. 
2. For TF-IDF tests, being disjunctive means that for every query question, the TF-IDF matrix is calculated independently(not including this question). All disjunctive tests take much more time than basic ones and usually give a couple of percent lower accuracy.
3. `weighted_qa` tests dependence on whether the word probabilities are taken from questions and/or answers corpuses. The results should not differ much.
4. `sw` and `lemm` parameters indicate using stop words removal and word lemmatization.
5. The cells of the result table are in the form: `f, s, t* : α`, where *f* is the accuracy with the first guess, *s* is the second guess, when the first guess was wrong and optionally *t* is same logic. α is the best α parameter for this test, taken out of some predefined range.
6. The Q78 data tests run 15-20 minutes. However, there are 3 times more questions in the new Q78 dataset and I am currently waiting for it to finish. As I can see, it will take approximately 3-4 hours. Disjunctive tests take a lot of time; the code is not optimized.


## Main files
* **faq50_adapted.py** contains most of the methods for creating and testing embeddings.
* **test_methods.py** tests different parameters for faq50 as described above.
* **tfidf_classifier.py** does TF-IDF operations. 
