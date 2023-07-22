from transformers import BertTokenizer, BertConfig, BertModel
import torch
import numpy as np
from cs_lemmatizer import LMTZR

import logging
logging.basicConfig(level=logging.INFO)

# print(torch.cuda.is_available())


class SlavicBERT:
    def __init__(self, 
                 dir_to_models, 
                 ):

        # Load the configuration file
        self.config = BertConfig.from_json_file(dir_to_models + "/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/config.json")

        # Load the model weights
        self.model = BertModel.from_pretrained(dir_to_models + "/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/pytorch_model.bin", 
                                          config=self.config,
                                          )

        # Tokenizer
        self.vocab_path = dir_to_models + "/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/vocab.txt"
        self.t_config = BertConfig.from_json_file(dir_to_models + "/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/tokenizer_config.json")
        self.tokenizer = BertTokenizer.from_pretrained(dir_to_models + "/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1",
                                          config=self.t_config, 
                                          vocab_file=self.vocab_path
                                          )

    def get_mean_sentence_embedding(self, sentence, sw=False, lm=False, mean=True):
        if sw or lm:
            sentence = " ".join(LMTZR.clean_corpus(sentence, rm_stop_words=sw, lemm=lm))

        tokenized_text = self.tokenizer.tokenize(sentence)
        # print(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text) # ids from vocab.txt
        tokens_tensor = torch.tensor(np.array([indexed_tokens]))
        self.model.eval()
        with torch.no_grad():
            last_hidden_states = self.model(tokens_tensor)
            embedding = last_hidden_states[0][0,:,:]
        if mean: # make one mean sentence embedding or leave word embeddings list
            embedding = np.array(torch.mean(embedding, dim=0))
            embedding /= np.linalg.norm(embedding)
            return embedding
        else:
            return embedding.numpy()
    



if __name__ == "__main__":

    dir_to_models = "models"
    m = SlavicBERT()

    # text = "[CLS] Машина гонит по тропе [SEP]"
    # text2 = "[CLS] Мотоцикл едет по дороге [SEP]"

    text = "[CLS] jak přihlásit do zahraničí [SEP]"
    text2 = "[CLS] pomoc s přihláškou užitného vzoru [SEP]"
    # text2 = "[CLS] Co se musí udělat pro prodloužení platnosti zápisu užitného vzoru [SEP]"

    embeddings = m.get_mean_sentence_embedding(text)
    embeddings2 = m.get_mean_sentence_embedding(text2)

    print("Cosine sim: ", embeddings @ embeddings2)


