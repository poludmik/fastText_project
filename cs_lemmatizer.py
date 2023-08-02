import simplemma
from simplemma import simple_tokenizer
from stop_words import get_stop_words

exclusion_list = ["kdo", "jak", "kolik", "co", "proč", "hodně", "který", "mezi", "chtít", "jeden", "mít",
                  "proti", "kde", "nebo", "kdy", "bez"]

class LMTZR:

    @staticmethod
    def lemmatize_cs(word):
        return simplemma.lemmatize(word, lang="cs")
    
    @staticmethod
    def tokenize(sentence):
        return simple_tokenizer(sentence.lower())
        # return sentence.lower().replace('\n', ' ').split()

    @staticmethod
    def remove_stop_words_from_sentence(sent_list: list[str]):
        return [w for w in sent_list if w not in get_stop_words('cz') or w in exclusion_list]
    
    @staticmethod
    def clean_corpus(corpus: str, rm_stop_words=True, lemm=True):
        "Tokenizes, lemmatizes and removes stop words from a sentence-corpus"
        # print('\n' + corpus)

        spec_symbols = LMTZR.has_a_tm_symbol(corpus)

        if lemm:
            words = [LMTZR.lemmatize_cs(w) for w in LMTZR.tokenize(corpus)]
        else:
            words = [w for w in LMTZR.tokenize(corpus)]

        if rm_stop_words:
            words = LMTZR.remove_stop_words_from_sentence(words)

        return words + spec_symbols

    @staticmethod
    def has_a_tm_symbol(document: str):
        symbols = []
        for c in document:
            if ord(c) == 174 or ord(c) == 169: # r in a circle or C in a circle
                symbols.append(c)
        return symbols