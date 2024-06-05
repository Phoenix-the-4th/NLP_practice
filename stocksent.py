import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# importing dataset
df = pd.read_csv("stock.csv", encoding='ISO-8859-1')
# print(df.head())

# splitting dataset
train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20141231']

# cleaning data
data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

# data acclimation
list1= [i for i in range(25)]
new_Index=[str(i) for i in list1]
data.columns= new_Index
# print(data.head(5))

# preprocessing
for index in new_Index:
    data[index]=data[index].str.lower()
# print(data.head(1))
' '.join(str(x) for x in data.iloc[1,0:25])
headlines = []
for row in range(0,len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row,0:25]))
# print(headlines[0])

# BOW to convert text data to numerical
countvector = CountVectorizer(ngram_range=(2,2))
traindataset = countvector.fit_transform(headlines)

#randomforest classifier
randomclassifier = RandomForestClassifier(n_estimators=200,criterion='entropy')
randomclassifier.fit(traindataset,train['Label'])

# predict on test dataset
test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row,2:27]))
test_dataset = countvector.transform(test_transform)
predictions = randomclassifier.predict(test_dataset)

#display results
matrix=confusion_matrix(test['Label'], predictions)
print(matrix)
score=accuracy_score(test['Label'], predictions)
print(score)
report=classification_report(test['Label'], predictions)
print(report)