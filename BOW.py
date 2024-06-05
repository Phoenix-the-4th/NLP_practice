import pandas as pd
import chardet
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


paragraph = """Wikipedia[c] is a free content online encyclopedia written and maintained by a community of volunteers, known as Wikipedians, through open collaboration and the wiki software MediaWiki. Wikipedia is the largest and most-read reference work in history,[3][4] and is consistently ranked among the ten most visited websites; as of April 2024, it was ranked fourth by Semrush,[5] and seventh by Similarweb.[6] Founded by Jimmy Wales and Larry Sanger on January 15, 2001, since 2003 Wikipedia has been hosted by the Wikimedia Foundation, an American nonprofit organization funded mainly by donations from readers.[7]

Initially only available in English, editions of Wikipedia in more than 300 other languages have been developed. The English Wikipedia, with its 6.8 million articles, is the largest of the editions, which together comprise more than 63 million articles and attract more than 1.5 billion unique device visits and 13 million edits per month (about 5 edits per second on average) as of April 2024.[W 1] Roughly 25% of Wikipedia's traffic is from the United States, followed by Japan at 5.8%, the United Kingdom at 5.7%, Germany at 5%, Russia at 4.9%, and the remaining 54% split among other countries, according to Similarweb.[8]

Wikipedia has been praised for its enablement of the democratization of knowledge, extent of coverage, unique structure, and culture. It has been criticized for exhibiting systemic bias, particularly gender bias against women and geographical bias against the Global South (Eurocentrism).[9][10] While the reliability of Wikipedia was frequently criticized in the 2000s, it has improved over time, receiving greater praise from the late 2010s onward,[3][11][12] while becoming an important fact-checking site.[13][14] Wikipedia has been censored by some national governments, ranging from specific pages to the entire site.[15][16] Articles on breaking news are often accessed as sources for frequently updated information about those events.[17][18]"""
ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()


# # checking encoding of file
# # with open('spam.csv', 'rb') as file:
# #     print(chardet.detect(file.read()))

# # getting data
# messages = pd.read_csv('C:\\Internship\\NLP_practice\\spam.csv', sep='\t', names=["label", "message"])

# # pre processing text
# stemmer = PorterStemmer()
# corpus=[]
# print(messages)
# for i in range(0,len(messages)):
#     review=re.sub('[^a-zA-z]',' ',messages['message'][i])
#     review=review.lower()
#     review=review.split()
#     review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
#     review=' '.join(review)
#     corpus.append(review)

# # BOW
# vectorizer=CountVectorizer(max_features=100,binary=True)
# X=vectorizer.fit_transform(corpus).toarray()
# np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

# # BOW ngram
# cv=CountVectorizer(max_features=100,binary=True,ngram_range=(2,3))
# X=cv.fit_transform(corpus).toarray()