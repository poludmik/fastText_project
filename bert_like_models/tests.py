# To test that I exctract mean embeddings the right way in Robeczech

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('ufal/robeczech-base')


text = "Co chrání autorské právo?"
# text2 = "Co se stane po podání stížnosti?"
text2 = "Kam se obrátit s dotazy k autorským právům?"

# text = "Auto jede po cestě do Ostravy"
# text2 = "Pomeranče jsou nejlepší ovoce"

emb = model.encode(text)
print(emb.shape)
emb2 = model.encode(text2)
print(emb2.shape)

print("Cosine:", emb @ emb2)






# There are max ~350 tokens in an answer
# import pandas as pd
# path_to_a = "upv_faq/data/FAQ76_answers.xlsx"

# df = pd.read_excel(path_to_a)
# column_data = df['answer'].tolist()
# max_len = 0
# max_a = ""
# for row in column_data:
#     if len(row.split()) > max_len:
#         max_len = len(row.split())
#         max_a = row.split()
# print(max_len)
# print(max_a)


