import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


messages = pd.read_csv('SMSSpamCollection.txt', sep = '\t', names = ["label", "message"])
stemmer = PorterStemmer()
corpus = []
root = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    root.append(' '.join([word for word in review if not word in stopwords.words('english')]))
    review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
c2 = corpus.copy()

vec = CountVectorizer(max_features=2500)
X = vec.fit_transform(corpus).toarray()
y = pd.get_dummies(messages["label"])
y = y.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)

######### check how good it
confusion_m = confusion_matrix(y_test, y_pred)
print(confusion_m)
accuracy_s = accuracy_score(y_test, y_pred)
print(accuracy_s)



######### lemmatized version
lemmer = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
vec = CountVectorizer(max_features=2500)
X = vec.fit_transform(corpus).toarray()
y = pd.get_dummies(messages["label"])
y = y.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)
confusion_m = confusion_matrix(y_test, y_pred)
print(confusion_m)
accuracy_s = accuracy_score(y_test, y_pred)
print(accuracy_s)


# with open("output.txt", "a") as f:
#     for i in range(len(corpus)):
#         if root[i] != corpus[i]:
#             print("############################################", file = f)
#         print(root[i], corpus[i], c2[i], sep = '\n', end = '\n\n', file = f)


######### tfidf versions of both
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [stemmer.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
vec = TfidfVectorizer(max_features=2500)
X = vec.fit_transform(corpus).toarray()
y = pd.get_dummies(messages["label"])
y = y.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)
confusion_m = confusion_matrix(y_test, y_pred)
print(confusion_m)
accuracy_s = accuracy_score(y_test, y_pred)
print(accuracy_s)

corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
vec = TfidfVectorizer(max_features=2500)
X = vec.fit_transform(corpus).toarray()
y = pd.get_dummies(messages["label"])
y = y.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
spam_detect_model = MultinomialNB().fit(X_train, y_train)
y_pred=spam_detect_model.predict(X_test)
confusion_m = confusion_matrix(y_test, y_pred)
print(confusion_m)
accuracy_s = accuracy_score(y_test, y_pred)
print(accuracy_s)


# CONCLUSION: stemming gave more accurate results than lemmatizing
# possible reasoning: data contains typos and words that do not exist in lemmatizer dictionary
# assuming that bow performing better than tf-idf for same reason: inconsistency in data