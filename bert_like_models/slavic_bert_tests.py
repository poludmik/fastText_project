from transformers import BertTokenizer, BertConfig, BertModel
import torch
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)

# print(torch.cuda.is_available())


# Load the configuration file
config = BertConfig.from_json_file("models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/config.json")

# Load the model weights
model = BertModel.from_pretrained("models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/pytorch_model.bin", 
                                  config=config,
                                  )

# Tokenizer
vocab_path = "models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/vocab.txt"
t_config = BertConfig.from_json_file("models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/tokenizer_config.json")
tokenizer = BertTokenizer.from_pretrained("models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1",
                                          config=t_config, 
                                          vocab_file=vocab_path,
                                          )




# text = "[CLS] Машина гонит по тропе [SEP]"
# text2 = "[CLS] Мотоцикл едет по дороге [SEP]"

text = "[CLS] jak přihlásit do zahraničí [SEP]"
text2 = "[CLS] pomoc s přihláškou užitného vzoru [SEP]"
# text2 = "[CLS] Co se musí udělat pro prodloužení platnosti zápisu užitného vzoru [SEP]"



# Get word embeddings for text:
tokenized_text = tokenizer.tokenize(text)
print(tokenized_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text) # ids from vocab.txt
tokens_tensor = torch.tensor(np.array([indexed_tokens]))
# segments_ids = [1] * len(tokenized_text)
# segments_tensors = torch.tensor([segments_ids])
model.eval()
with torch.no_grad():
    last_hidden_states = model(tokens_tensor)
    embeddings = last_hidden_states[0][0,:,:]
    print(embeddings.shape)


# Get word embeddings for text2:
tokenized_text = tokenizer.tokenize(text2)
print(tokenized_text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
tokens_tensor = torch.tensor(np.array([indexed_tokens]))
model.eval()
with torch.no_grad():
    last_hidden_states = model(tokens_tensor)
    embeddings2 = last_hidden_states[0][0,:,:]
    print(embeddings2.shape)



# Compute cosine similarity of mean embeddings
embeddings = np.array(torch.mean(embeddings, dim=0))
embeddings /= np.linalg.norm(embeddings)
embeddings2 = np.array(torch.mean(embeddings2, dim=0))
embeddings2 /= np.linalg.norm(embeddings2)
print("Cosine sim: ", embeddings @ embeddings2)


