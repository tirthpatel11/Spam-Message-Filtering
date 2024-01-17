import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd
import csv

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

df=pd.read_csv('unfiltered.csv', encoding="ISO-8859-1")
df.drop(columns=['v1','Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)

for i in range(len(df)):
    # 1. preprocess
    transformed_sms = transform_text(df['v2'][i])
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 0:
        df1 = pd.DataFrame({df['v2'][i]})
        df1.to_csv('filtered.csv', mode='a', header=False)
    