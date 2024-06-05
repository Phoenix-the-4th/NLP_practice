from tensorflow.keras.preprocessing.text import one_hot # type: ignore
from tensorflow.keras.layers import Embedding # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
import numpy as np

sent=['the glass of milk', 'the glass of juice', 'the cup of tea', 'I am a good boy', 'I am a good developer', 'understand the meaning of words', 'your videos are good',]
voc_size = 10000
onehot_repr = [one_hot(words, voc_size) for words in sent]
sent_length = 8
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
# print(embedded_docs)
dim = 10
model = Sequential()
model.add(Embedding(voc_size, dim))
model.compile('adam', 'mse')
print(model.summary())
X = model.predict(embedded_docs)