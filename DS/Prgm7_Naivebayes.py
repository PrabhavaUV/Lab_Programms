import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("Data/Placement_Data.csv")

print(df.columns)

X=df[["ssc_p","hsc_p","degree_p","etest_p","mba_p"]]
y=df["status"]

le=LabelEncoder()
y=le.fit_transform(y)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

gnb=GaussianNB()
gnb.fit(X_train,y_train)
print("Gaussian Naive Bayes:", gnb.score(X_test,y_test))
g_predict=gnb.predict(X_test)
print(confusion_matrix(y_test,g_predict))
print("Gaussian Naive Bayes:", classification_report(y_test,g_predict))

mnb=MultinomialNB()
mnb.fit(X_train,y_train)
print("Multinomial Naive Bayes:", mnb.score(X_test,y_test))
m_predict=mnb.predict(X_test)
print(confusion_matrix(y_test,m_predict))
print("Multinomial Naive Bayes:", classification_report(y_test,m_predict))

bnb=BernoulliNB()
bnb.fit(X_train,y_train)
print("Bernoulli Naive Bayes:", bnb.score(X_test,y_test))
b_predict=bnb.predict(X_test)
print(confusion_matrix(y_test,b_predict))
print("Bernoulli Naive Bayes:", classification_report(y_test,b_predict))