import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

df = pd.read_csv('Data/student_performance_new.csv')
print(df.columns)
print(df.info())

X = df[["Test Result ","Quiz Result ","Assignment Result "]]
y = df.Result

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
clf = DecisionTreeClassifier(criterion='gini',splitter='random',random_state=42,max_depth=5)
clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test,y_pred))

plt.figure(figsize=(10, 8))
plot_tree(clf, filled=True, feature_names=X.columns.tolist(), class_names=['Fail', 'Pass'])
plt.title("Decision Tree for Student Performance")
plt.show()
