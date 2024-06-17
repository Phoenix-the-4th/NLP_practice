import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np

from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords

from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score



df = pd.read_csv('fake-news/train.csv')
df = df.dropna()
X = df.drop('label', axis = 1)
y = df['label']
# df = df.dropna()
messages = X.copy()
messages.reset_index(inplace = True)


corpus = []
stemmer = PorterStemmer()
for i in range(len(messages)):
    line = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    line = line.lower()
    line = line.split()
    line = [stemmer.stem(word) for word in line if word not in stopwords.words('english')]
    line = ' '.join(line)
    corpus.append(line)

vocab = 5000
onehot = [one_hot(line, vocab) for line in corpus]
sentence = 20
docs = pad_sequences(onehot, padding = 'pre', maxlen = sentence)

features = 40
model = Sequential()
model.add(Embedding(vocab, features))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

X_final = np.array(docs)
y_final = np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size=64)

y_pred = [0 if cl < 0.5 else 1 for cl in model.predict(X_test)]
print(model.summary())
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))