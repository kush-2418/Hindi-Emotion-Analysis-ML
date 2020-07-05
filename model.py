#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:22:00 2020

@author: kush
"""



import os
import pandas as pd
import re
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from googletrans import Translator
import pickle


%matplotlib qt5

from sklearn.feature_extraction.text import CountVectorizer


file_content = []
dir1 = '/Users/kush/Downloads/Hindi emotion detection/angry'
with os.scandir(dir1) as entries:
    for entry in entries:
        if entry.is_file():
            with open(entry,'r') as data:
                file_content.append(data.read())

angry_df = pd.DataFrame(file_content,columns=['Text'])
angry_df['Emotion'] = 0
angry_df.to_csv('angry.csv',encoding='utf-8')


file_content = []
dir1 = '/Users/kush/Downloads/Hindi emotion detection/happy'
with os.scandir(dir1) as entries:
    for entry in entries:
        if entry.is_file():
            with open(entry,'r') as data:
                file_content.append(data.read())

happy_df = pd.DataFrame(file_content,columns=['Text'])
happy_df['Emotion'] = 1
happy_df.to_csv('happy.csv',encoding='utf-8')

file_content = []
dir1 = '/Users/kush/Downloads/Hindi emotion detection/sad'
with os.scandir(dir1) as entries:
    for entry in entries:
        if entry.is_file():
            with open(entry,'r') as data:
                file_content.append(data.read())

sad_df = pd.DataFrame(file_content,columns=['Text'])
sad_df['Emotion'] = 2
sad_df.to_csv('sad.csv',encoding='utf-8')

file_content = []
dir1 = '/Users/kush/Downloads/Hindi emotion detection/neutral'
with os.scandir(dir1) as entries:
    for entry in entries:
        if entry.is_file():
            with open(entry,'r') as data:
                file_content.append(data.read())

neutral_df = pd.DataFrame(file_content,columns=['Text'])
neutral_df['Emotion'] = 3
neutral_df.to_csv('neutral.csv',encoding='utf-8')


angry_df = pd.read_csv("/Users/kush/Downloads/Hindi emotion detection/DataCSV/angry.csv",encoding='utf-8')
happy_df = pd.read_csv("/Users/kush/Downloads/Hindi emotion detection/DataCSV/happy.csv",encoding='utf-8')
sad_df = pd.read_csv("/Users/kush/Downloads/Hindi emotion detection/DataCSV/sad.csv",encoding='utf-8')
neutral_df = pd.read_csv("/Users/kush/Downloads/Hindi emotion detection/DataCSV/neutral.csv",encoding='utf-8')


angry_df = angry_df.drop(['index'],axis=1)
happy_df = happy_df.drop(['index'],axis=1)
sad_df = sad_df.drop(['index'],axis=1)
neutral_df = neutral_df.drop(['index'],axis=1)

df = pd.concat([angry_df,happy_df,sad_df,neutral_df],axis=0)
df
df = df.reset_index()
df = df.drop(['index'],axis=1)
df.to_csv('emotion.csv',encoding='utf-8',index=None)


df['Emotion'].value_counts()



f=codecs.open("/Users/kush/Downloads/Hindi emotion detection/hindi_stopwords.txt",encoding='utf-8')

stopwords=[x.strip() for x in f.readlines()]
# wordcloud

plt.figure(figsize = (20,20)) 
wc = WordCloud(font_path='Lohit-Devanagari.ttf',max_words = 300 , width = 300 , height = 600,stopwords = stopwords,colormap='Reds').generate(" ".join(df[df.Emotion == 0].Text))
plt.imshow(wc , interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

plt.figure(figsize = (20,20)) 
wc = WordCloud(font_path='Lohit-Devanagari.ttf',max_words = 300 , width = 300 , height = 600,stopwords = stopwords,colormap='Greens').generate(" ".join(df[df.Emotion == 1].Text))
plt.imshow(wc , interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


plt.figure(figsize = (20,20))
wc = WordCloud(font_path='Lohit-Devanagari.ttf',max_words = 300 , width = 300 , height = 600,stopwords = stopwords,colormap='Purples').generate(" ".join(df[df.Emotion == 2].Text))
plt.imshow(wc , interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()


plt.figure(figsize = (20,20)) 
wc = WordCloud(font_path='Lohit-Devanagari.ttf',max_words = 300 , width = 300 , height = 600,stopwords = stopwords,colormap='spring').generate(" ".join(df[df.Emotion == 3].Text))
plt.imshow(wc , interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()



def remove_stopwords(text):
    text = [t for t in text.split() if t not in stopwords]
    text = " ".join(text)
    return text
    

df['Text'] = df['Text'].apply(lambda x: remove_stopwords(x))


from sklearn.feature_extraction.text import CountVectorizer



def tokenize(i):
    return i.split(' ')

X = df['Text'].values
y = df['Emotion'].values

cv = CountVectorizer(min_df=2, ngram_range=(1, 3), encoding='utf-8',tokenizer=tokenize)
cv.fit(X)
X_vect = cv.transform(X)
print(cv.get_feature_names())
pickle.dump(cv, open('/Users/kush/Downloads/Hindi emotion detection/hindi_cv.pkl', 'wb'))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.1, random_state=48,stratify=y)


from sklearn.model_selection import  GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Create param grid.

import numpy as np

param_grid = [
    {
     'penalty' : [ 'l2'],
    'C' : np.logspace(-4, 4, 20),
    'solver' : ['liblinear','newton-cg','sag','saga'],
    'max_iter' : [100,1000,2500,5000]}
]


lr = LogisticRegression()

# Create grid search object

clf = GridSearchCV(lr, param_grid = param_grid, cv = 10, verbose=True, n_jobs=-1)
clf.get_params().keys()


'''
{'C': 0.615848211066026,
 'class_weight': None,
 'dual': False,
 'fit_intercept': True,
 'intercept_scaling': 1,
 'l1_ratio': None,
 'max_iter': 2500,
 'multi_class': 'auto',
 'n_jobs': None,
 'penalty': 'l2',
 'random_state': None,
 'solver': 'saga',
 'tol': 0.0001,
 'verbose': 0,
 'warm_start': False}
'''


# Fit on data

best_clf = clf.fit(X_train, y_train)

pickle.dump(best_clf, open('/Users/kush/Downloads/Hindi emotion detection/hindi_model.pkl', 'wb'))
best_clf.best_estimator_.get_params()
print('Model accuracy is',best_clf.score(X_test, y_test))

y_pred = best_clf.predict(X_test)
    
accuracy = metrics.accuracy_score(y_pred, y_test)
precision = metrics.precision_score(y_pred, y_test,average='weighted')        
f1_score = metrics.f1_score(y_pred, y_test,average='weighted')    
recall = metrics.recall_score(y_pred, y_test,average='weighted')


print('Test accuracy = ', accuracy)
print('Test precision = ', precision)
print('Test f1-score = ', f1_score)
print('Test recall = ', recall)

def predict_sarcasm(sample_review):
    sample_review = [sample_review]
    test_sample = cv.transform(sample_review)  
    pred = best_clf.predict(test_sample)
    return pred

import random
sample_text = random.choice(df[["Text",'Emotion']].values.tolist())

trans =  Translator()
text = input("Enter the text ")

t = trans.translate(text,src='en',dest='hi')

print(f'Source: {t.src}')
print(f'Dest : {t.dest}')
print(f'Dest : {t.text}')


prediction = predict_sarcasm(t.text)

prediction  = int(prediction)
if prediction ==0:
    prediction = 'Angry'
elif prediction == 1:
    prediction ='Happy'
elif prediction == 2:
    prediction = 'Sad'
else:
    prediction = 'Neutral'
print('The emotion behind the sentence "{}" is {}: '.format(t.text,prediction))

sample_text = "जो हाथ सेवा के लिए उठते है, वे प्रार्थना करते होंठों से पवित्र है!"
prediction = predict_sarcasm(sample_text)

prediction  = int(prediction)
if prediction ==0:
    prediction = 'Angry'
elif prediction == 1:
    prediction ='Happy'
elif prediction == 2:
    prediction = 'Sad'
else:
    prediction = 'Neutral'
print('The emotion behind the sentence "{}" is {}: '.format(sample_text,prediction))

sample_text ="उस दिन फातिमा से अलग होते हुए लड़के का मन वहुत दुखी था"
prediction = predict_sarcasm(sample_text)

prediction  = int(prediction)
if prediction ==0:
    prediction = 'Angry'
elif prediction == 1:
    prediction ='Happy'
elif prediction == 2:
    prediction = 'Sad'
else:
    prediction = 'Neutral'
print('The emotion behind the sentence "{}" is {}: '.format(sample_text,prediction))

sample_text =u"ज्यादा आवेश में आने की जरूरत नहीं है"
prediction = predict_sarcasm(sample_text)

prediction  = int(prediction)
if prediction ==0:
    prediction = 'Angry'
elif prediction == 1:
    prediction ='Happy'
elif prediction == 2:
    prediction = 'Sad'
else:
    prediction = 'Neutral'
print('The emotion behind the sentence "{}" is {}: '.format(sample_text,prediction))