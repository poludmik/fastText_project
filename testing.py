import fasttext
import vec_math

# model = fasttext.train_unsupervised('../data/fil9')
# model = fasttext.train_unsupervised('data/fil9', minn=2, maxn=5, dim=300)
# model.save_model("models/fil9.bin")

model = fasttext.load_model("models/fil9.bin")

x = "metallica"
y = "abba"
vec1 = model.get_word_vector(x)
vec2 = model.get_word_vector(y)

print(model.get_word_vector(x).shape)

print(model.get_nearest_neighbors(x))

print(model.get_analogies("berlin", "germany", "czechia"))

print("scipy:", vec_math.cosine_sim_scipy(vec1, vec2))
print("numpy:", vec_math.cosine_sim_numpy(vec1, vec2))

