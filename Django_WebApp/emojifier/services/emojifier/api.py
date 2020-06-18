import emoji
import numpy as np
import pandas as pd
from keras.models import model_from_json

with open("services/emojifier/Web_Model.json", "r") as file:
    model = model_from_json(file.read())

model.load_weights("services/emojifier/Web_Model.h5")
model._make_predict_function()

emoji_dictionary = {"0": "\u2764\uFE0F", "1": ":baseball:", "2": ":beaming_face_with_smiling_eyes:", "3": ":downcast_face_with_sweat:",  "4": ":fork_and_knife:" }

embeddings = {}
with open('services/emojifier/glove.6B.50d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coeffs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coeffs

def getOutputEmbeddings(X):

    embedding_matrix_output = np.zeros((X.shape[0], 10, 50))
    for ix in range(X.shape[0]):
        X[ix] = X[ix].split()
        for jx in range(len(X[ix])):
            embedding_matrix_output[ix][jx] = embeddings[X[ix][jx].lower()]

    return embedding_matrix_output

def predict(x):
    X = pd.Series([x])
    emb_X = getOutputEmbeddings(X)
    p = model.predict_classes(emb_X)
    return emoji.emojize(emoji_dictionary[str(p[0])])
