import simplemma
from simplemma import simple_tokenizer
from stop_words import get_stop_words



class LMTZR:

    @staticmethod
    def lemmatize_cs(word):
        return simplemma.lemmatize(word, lang="cs")
    
    @staticmethod
    def tokenize(sentence):
        return simple_tokenizer(sentence)

    @staticmethod
    def remove_stop_words_from_sentence(sent_list: list[str]):
        return [w for w in sent_list if w not in get_stop_words('cz')]
    
    @staticmethod
    def clean_corpus(corpus: str, rm_stop_words=True):
        "Tokenizes, lemmatizes and removes stop words from a sentence-corpus"
        words = [LMTZR.lemmatize_cs(w) for w in LMTZR.tokenize(corpus)]
        if not rm_stop_words:
            return words
        return LMTZR.remove_stop_words_from_sentence(words)