from transformers import BertTokenizer, BertConfig, BertModel
import torch

import logging
logging.basicConfig(level=logging.INFO)

# print(torch.cuda.is_available())


# Load the configuration file
config = BertConfig.from_json_file("models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/config.json")

# Load the model weights
model = BertModel.from_pretrained("models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/pytorch_model.bin", 
                                  config=config, 
                                  output_hidden_states=True
                                  )
model.eval()

# Tokenize the question
vocab_path = "models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/vocab.txt"
t_config = BertConfig.from_json_file("models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1/tokenizer_config.json")
tokenizer = BertTokenizer.from_pretrained("models/bg_cs_pl_ru_cased_L-12_H-768_A-12_pt_v1",
                                          config=t_config, 
                                          vocab_file=vocab_path
                                          )

# encoded_question = tokenizer(question, return_tensors='pt')

# encoded_dict = tokenizer.encode_plus(
#                         question,                      # Sentence to split into tokens
#                         add_special_tokens = True, # Add special token '[CLS]' and '[SEP]'
#                         max_length = 64,           # Pad & truncate all sentences.
#                         padding = 'longest',
#                         return_attention_mask = True,   # Construct attention masks.
#                         return_tensors = 'pt',     # Return pytorch tensors.
#                         truncation=True,
#                    )


text = "[CLS] Jak patentovat právo изобретение. [SEP]"

# Split the sentence into tokens.
tokenized_text = tokenizer.tokenize(text)

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [1] * len(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

with torch.no_grad():

    outputs = model(indexed_tokens, segments_tensors)

    hidden_states = outputs[2]

# print(encoded_dict)

# output = model(encoded_dict)



