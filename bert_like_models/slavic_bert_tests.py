from transformers import BertTokenizer, BertConfig, BertModel
import torch
import numpy as np
from cs_lemmatizer import LMTZR

# import logging
# logging.basicConfig(level=logging.INFO)
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

        # print(self.tokenizer.tokenize("Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Nullam justo enim, consectetuer nec, ullamcorper ac, vestibulum in, elit. In convallis. Nullam lectus justo, vulputate eget mollis sed, tempor sed magna. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Nulla non lectus sed nisl molestie malesuada. Maecenas lorem. Praesent vitae arcu tempor neque lacinia pretium. Integer malesuada. Praesent id justo in neque elementum ultrices. Temporibus autem quibusdam et aut officiis debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint et molestiae non recusandae. Pellentesque arcu"
        #                     +" Integer rutrum, orci vestibulum ullamcorper ultricies, lacus quam ultricies odio, vitae placerat pede sem sit amet enim. Donec quis nibh at felis congue commodo. Nulla quis diam. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum. Phasellus enim erat, vestibulum vel, aliquam a, posuere eu, velit. In enim a arcu imperdiet malesuada. Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Suspendisse sagittis ultrices augue. Donec vitae arcu. In convallis. Duis ante orci, molestie vitae vehicula venenatis, tincidunt ac pede. Duis sapien nunc, commodo et, interdum suscipit, sollicitudin et, dolor. Fusce aliquam vestibulum ipsum. Suspendisse sagittis ultrices augue. Pellentesque sapien. Sed ac dolor sit amet purus malesuada congue."
        #                     +" Mauris elementum mauris vitae tortor. Vestibulum fermentum tortor id mi. Etiam posuere lacus quis dolor. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Maecenas sollicitudin. Duis sapien nunc, commodo et, interdum suscipit, sollicitudin et, dolor. Vestibulum erat nulla, ullamcorper nec, rutrum non, nonummy ac, erat. Fusce suscipit libero eget elit. In laoreet, magna id viverra tincidunt, sem odio bibendum justo, vel imperdiet sapien wisi sed libero. Duis viverra diam non justo. Nam sed tellus id magna elementum tincidunt. Nulla accumsan, elit sit amet varius semper, nulla mauris mollis quam, tempor suscipit diam nulla vel leo. Mauris elementum mauris vitae tortor. Maecenas fermentum, sem in pharetra pellentesque, velit turpis volutpat ante, in pharetra metus odio a lectus. Integer pellentesque quam vel velit. Integer vulputate sem a nibh rutrum consequat."
        #                     +" Nullam faucibus mi quis velit. Integer vulputate sem a nibh rutrum consequat. Aliquam ornare wisi eu metus. Fusce tellus. Nullam feugiat, turpis at pulvinar vulputate, erat libero tristique tellus, nec bibendum odio risus sit amet ante. Phasellus faucibus molestie nisl. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos hymenaeos. Etiam egestas wisi a erat. Nam sed tellus id magna elementum tincidunt. Pellentesque sapien. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam quaerat voluptatem. Duis sapien nunc, commodo et, interdum suscipit, sollicitudin et, dolor. Donec iaculis gravida nulla. Nulla pulvinar eleifend sem. Nunc dapibus tortor vel mi dapibus sollicitudin. Duis ante orci, molestie vitae vehicula venenatis, tincidunt ac pede. Etiam posuere lacus quis dolor."
        #                     +" Integer in sapien. In convallis. Cras elementum. Aenean fermentum risus id tortor. Pellentesque pretium lectus id turpis. Nemo enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Nullam eget nisl. Aenean id metus id velit ullamcorper pulvinar. Mauris suscipit, ligula sit amet pharetra semper, nibh ante cursus purus, vel sagittis velit mauris vel metus. Sed ac dolor sit amet purus malesuada congue. Praesent dapibus. Integer rutrum, orci vestibulum ullamcorper ultricies, lacus quam ultricies odio, vitae placerat pede sem sit amet enim. In dapibus augue non sapien. Nullam lectus justo, vulputate eget mollis sed, tempor sed magna. Pellentesque ipsum. Integer in sapien. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Phasellus rhoncus. Nulla est."))

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
    

