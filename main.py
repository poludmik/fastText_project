import fasttext

print("kek")

# model = fasttext.train_unsupervised('../data/fil9')
# model = fasttext.train_unsupervised('data/fil9', minn=2, maxn=5, dim=300)
# model.save_model("result/fil9.bin")

model = fasttext.load_model("result/fil9.bin")

x = "slipknot"
print(model.get_word_vector(x).shape)

print(model.get_nearest_neighbors(x))

print(model.get_analogies("berlin", "germany", "czechia"))

