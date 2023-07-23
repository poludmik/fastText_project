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
