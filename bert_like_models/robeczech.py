from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from cs_lemmatizer import LMTZR


class Robeczech(): # RoBERTa architecrute

    def __init__(self,
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained("ufal/robeczech-base")
        self.model = AutoModel.from_pretrained("ufal/robeczech-base")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        # print(self.model)

    def get_mean_sentence_embedding(self, sentence, sw=False, lm=False, mean=True):

        if sw or lm:
            sentence = " ".join(LMTZR.clean_corpus(sentence, rm_stop_words=sw, lemm=lm))

        tokenized_text = self.tokenizer.tokenize(sentence)
        # print(tokenized_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text) # ids from vocab.txt
        tokens_tensor = torch.tensor(np.array([indexed_tokens]))
        # print(tokens_tensor)

        # Or just encode (automatically adds [CLS] and [SEP] tokens)
        tokens_tensor = self.tokenizer.encode(sentence, return_tensors='pt') 
        # print(encoded)

        tokens_tensor = tokens_tensor.to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(input_ids=tokens_tensor)
            # print(out.pooler_output.shape) # tensor output only for [CLS] classification token
            embedding = out.last_hidden_state[0,:,:]
        if mean: # make one mean sentence embedding or leave word embeddings list

            embedding = np.array(torch.mean(embedding, dim=0).cpu())
            # embedding = np.array(embedding[0, :].cpu())
            # embedding = out.pooler_output.cpu()[0, :].numpy()

            embedding /= np.linalg.norm(embedding)

            return embedding
        else:
            return embedding.cpu().numpy() # return n word embeddings
