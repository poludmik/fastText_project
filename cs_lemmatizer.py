import simplemma
from simplemma import simple_tokenizer


class LMTZR:

    @staticmethod
    def lemmatize_cs(word):
        return simplemma.lemmatize(word, lang="cs")
    
    @staticmethod
    def tokenize(sentence):
        return simple_tokenizer(sentence)

