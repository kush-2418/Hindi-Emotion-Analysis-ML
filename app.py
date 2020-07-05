#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 20:27:17 2020

@author: kush
"""

## import libraries

import streamlit as st
import codecs
import pandas as pd
import unidecode
import re
from PIL import Image
import pickle
from wordcloud import WordCloud,STOPWORDS
from googletrans import Translator
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score,recall_score




def main():
    
    # Load Hindi stopwords
    
    f=codecs.open("/Users/kush/Downloads/Hindi emotion detection/hindi_stopwords.txt",encoding='utf-8')
    stopwords=[x.strip() for x in f.readlines()]
    
    # Load dataset
    
    @st.cache(allow_output_mutation=True)
    def load_dataset():
        df = pd.read_csv('/Users/kush/Downloads/Hindi emotion detection/DataCSV/emotion.csv',encoding='utf-8')
        return df
    
        pass
    
    # Wordcloud function
    def wordCloud(df,emotion):
        if emotion == 'Angry':
            num = 0
            color = 'Reds'
        elif emotion == 'Happy':
            num=1
            color = 'Greens'    
        elif emotion == 'Sad':
            num=2
            color = 'Purples'
        else:
            num=3
            color = 'spring'
    

        plt.figure(figsize = (8,8)) 
        wc = WordCloud(font_path='Lohit-Devanagari.ttf',width=1400, height=1300,stopwords = stopwords,colormap='Reds',collocations=False).generate(" ".join(df[df.Emotion == num].Text))
        plt.imshow(wc , interpolation = 'bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig('worldCloud.jpg')
        img = Image.open("worldCloud.jpg")
        return img
    
    
    def remove_stopwords(text):
        text = [t for t in text.split() if t not in stopwords]
        text = " ".join(text)
        return text
    
    def count_vect(df):
        
        df['Text'] = df['Text'].apply(lambda x: remove_stopwords(x))
        X = df['Text'].values
        y = df['Emotion'].values
        def tokenize(i):
            return i.split(' ')        
        
        cv = CountVectorizer(min_df=2, ngram_range=(1, 3), encoding='utf-8',tokenizer=tokenize)
        cv.fit(X) 
        X_vector = cv.transform(X)  # Getting Bag of words representation for all the documents
        X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.1, random_state=48,stratify=y)
        return X_train, X_test, y_train, y_test,cv

            
    def predict_emotion(model,cv,text):
        text = [text]
        cvect = cv.transform(text).toarray()
        pred = model.predict(cvect)
        return pred
    
    def translate(text):
        trans =  Translator()
        t = trans.translate(text,src='en',dest='hi')
        return t.src, t.dest, t.text
    
    
    
    st.markdown("<body style='background-color:white;'><h1 style='text-align: center; color: blue;'>Hindi Emotion Analysis</h1></body>", unsafe_allow_html=True)
    st.markdown("<body style='background-color:white;'><h3 style='text-align: center; color: green;'>SELECT YOUR ACTIVITIES FROM THE SIDEBAR 👈</h3></body>", unsafe_allow_html=True)

    if st.checkbox('Show data'):
        df = load_dataset()
        st.dataframe(df)
    st.sidebar.subheader("Perform the following task")
    
    select = ['Select','Word Cloud','Run Your Model','Run Pretrained Model']
    option = st.sidebar.selectbox("",select)
    
    if option==select[0]:
        pass
    
    # WordCloud
    elif option == select[1]:
        st.markdown("<body style='background-color:white;'><h2 style='text-align: center; color: orange;'>Word Cloud For</h2></body>", unsafe_allow_html=True)

        emotion = st.radio(' ',('Angry','Happy','Sad','Neutral'))
        df = load_dataset()
        img = wordCloud(df,emotion)
        st.image(img)
        
    # Train your model
    elif option == select[2]:
        
        classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine","Logistic Regression","Random Forest"))
        
        if classifier =="Support Vector Machine":
            df = load_dataset()
            X_train, X_test, y_train, y_test,cv  = count_vect(df)
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key='C')
            kernel = st.sidebar.radio("kernel",("rbf","linear"),key='kernel')
            gamma = st.sidebar.radio("Gamma (Kernel Coefficientt)", ("scale","auto"),key='gamma')
            if st.sidebar.button("Classify",key='classify1'):
                st.markdown("<body style='background-color:white;'><h2 style='text-align: center; color: orange;'>Support Vector Machine Results</h2></body>", unsafe_allow_html=True)

              
                model = SVC(C=C,kernel=kernel,gamma=gamma)
                model.fit(X_train,y_train)
                accuracy = model.score(X_test,y_test)
                y_pred = model.predict(X_test)
                precision = metrics.precision_score(y_pred, y_test,average='weighted')        
                f1_score = metrics.f1_score(y_pred, y_test,average='weighted')    
                recall = metrics.recall_score(y_pred, y_test,average='weighted')
                st.success("Accuracy --->   {} % ".format(accuracy.round(5)*100))                                
                st.success("Precision--->   {}".format(precision.round(2)))
                st.success("Recall --->   {}".format(recall.round(2)))
                st.success("F1-score --->   {} ".format(f1_score.round(2)))

                
                
        if classifier == "Logistic Regression":
            df = load_dataset()
            X_train, X_test, y_train, y_test,cv  = count_vect(df)
            st.sidebar.subheader("Model Hyperparameters")
            C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step=0.01,key='C_LR')
            max_iter = st.sidebar.slider("Maximum number of iterations",100,500,key='max_iter')
            if st.sidebar.button("Classify",key='classify2'):
                st.markdown("<body style='background-color:white;'><h2 style='text-align: center; color: orange;'>Logistic Regression Results</h2></body>", unsafe_allow_html=True)
                model = LogisticRegression(C=C,max_iter=max_iter)
                model.fit(X_train,y_train)
                accuracy = model.score(X_test,y_test)
                y_pred = model.predict(X_test)
                precision = metrics.precision_score(y_pred, y_test,average='weighted')        
                f1_score = metrics.f1_score(y_pred, y_test,average='weighted')    
                recall = metrics.recall_score(y_pred, y_test,average='weighted')
                st.success("Accuracy --->   {} % ".format(accuracy.round(5)*100))                                
                st.success("Precision--->   {}".format(precision.round(2)))
                st.success("Recall --->   {}".format(recall.round(2)))
                st.success("F1-score --->   {} ".format(f1_score.round(2)))

                
                   
                
        if classifier == "Random Forest":
            df = load_dataset()
            X_train, X_test, y_train, y_test,cv  = count_vect(df)
            st.sidebar.subheader("Model Hyperparameters")
            n_estim = st.sidebar.number_input("The number of trees in forest",100,5000,step=10,key='nest')
            max_depth = st.sidebar.number_input("The max depth of the trees",1,20,step=1,key='mdepth')
            if st.sidebar.button("Classify",key='classify3'):
                st.markdown("<body style='background-color:white;'><h2 style='text-align: center; color: orange;'>Random Forest Results</h2></body>", unsafe_allow_html=True)
                model = RandomForestClassifier(n_estimators=n_estim,max_depth=max_depth)
                model.fit(X_train,y_train)
                accuracy = model.score(X_test,y_test)
                y_pred = model.predict(X_test)
                precision = metrics.precision_score(y_pred, y_test,average='weighted')        
                f1_score = metrics.f1_score(y_pred, y_test,average='weighted')    
                recall = metrics.recall_score(y_pred, y_test,average='weighted')
                st.success("Accuracy --->   {} % ".format(accuracy.round(5)*100))                                
                st.success("Precision--->   {}".format(precision.round(2)))
                st.success("Recall --->   {}".format(recall.round(2)))
                st.success("F1-score --->   {} ".format(f1_score.round(2)))
                
    # Train pre-trained model 
    else:
        model = pickle.load(open('/Users/kush/Downloads/Hindi emotion detection/hindi_model.pkl','rb'))
        df = load_dataset()
        X_train, X_test, y_train, y_test,cv  = count_vect(df)
        accuracy = model.score(X_test,y_test)
        y_pred = model.predict(X_test)
        precision = metrics.precision_score(y_pred, y_test,average='weighted')        
        f1_score = metrics.f1_score(y_pred, y_test,average='weighted')    
        recall = metrics.recall_score(y_pred, y_test,average='weighted')
        st.markdown("<body style='background-color:white;'><h2 style='text-align: center; color: orange;'>Pre-trained Model Results</h2></body>", unsafe_allow_html=True)

        st.success("Accuracy --->   {} % ".format(accuracy.round(5)*100))                                
        st.success("Precision--->   {}".format(precision.round(2)))
        st.success("Recall --->   {}".format(recall.round(2)))
        st.success("F1-score --->   {} ".format(f1_score.round(2)))  
        
        # Predict on New Text
        
        st.markdown("<body style='background-color:blue;'><h2 style='text-align: center; color: white;'> Predict on New Text</h2></body>", unsafe_allow_html=True)
        st.markdown("<body style='background-color:white;'><h3 style='text-align: center; color: red;'> Do You Know Hindi ????</h3></body>", unsafe_allow_html=True)

        sel = ['Select','Yes, I Know',"No, I don't Know"]
        yn = st.selectbox("",sel)
        if yn == 'Select':
            pass
        elif yn == 'Yes, I Know':
            st.markdown("<body style='background-color:white;'><h3 style='text-align: center; color: brown;'> Great, Now enter the text in Hindi to predict the emotion</h3></body>", unsafe_allow_html=True)

            text = st.text_input("")
            if st.button('Predict Emotion'):
                pred = predict_emotion(model,cv,text)
                prediction  = int(pred)
                if prediction ==0:
                    prediction = 'Angry'
                elif prediction == 1:
                    prediction ='Happy'
                elif prediction == 2:
                    prediction = 'Sad'
                else:
                    prediction = 'Neutral'
                st.success('The emotion behind the sentence "{}" is {}: '.format(text,prediction))
        else:
            st.markdown("<body style='background-color:white;'><h3 style='text-align: center; color: brown;'> Don't Worry, enter the text in English to translate it in Hindi</h3></body>", unsafe_allow_html=True)

            st.markdown("#### ")
            st.write("Please enter a long sentence for better prediction")
            text = st.text_input("")
            if st.button('Predict Emotion'):
                src,dest, tex = translate(text)

                st.info("Your entered text is translated to {}".format(tex))

                pred = predict_emotion(model,cv,tex)
                prediction  = int(pred)
                if prediction ==0:
                    prediction = 'Angry 😡'
                elif prediction == 1:
                    prediction ='Happy 😀'
                elif prediction == 2:
                    prediction = 'Sad 😔'
                else:
                    prediction = 'Neutral 😐'
                st.success('The emotion behind the sentence "{}" is {}: '.format(tex,prediction))    
    
    

if __name__ == '__main__':
    main()