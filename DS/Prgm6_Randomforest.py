import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

df=pd.read_excel("Data/random_forest_dataset (1).xlsx",header=1)

print(df.columns)
df.dropna(subset=['Grade'], inplace=True)
x = df.drop(['Sl No ',"USN ","Name ","Title ","Grade"],axis=1)
y = df.Grade

print(x.head())

le=LabelEncoder()
y=le.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train,y_train)

feature_importances = pd.Series(rf.feature_importances_, index=x.columns).sort_values(ascending=False)
print(feature_importances)
plt.figure(figsize=(10, 6))
feature_importances.plot(kind='bar')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

param_grid = {
    'n_estimators': [100, 200, 300],  
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]  
}

grid_search = GridSearchCV(estimator=rf, param_grid = param_grid, cv=5, n_jobs=-1,verbose=1)

grid_search.fit(x_train,y_train)

bp=grid_search.best_params_
print("Best Parameters: ",bp)

best_rf = RandomForestClassifier(random_state=42,**bp)
best_rf.fit(x_train,y_train)
y_pred = best_rf.predict(x_test)

print("\n Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


plt.figure(figsize=(20,20))
plot_tree(best_rf.estimators_[5], feature_names = x.columns,class_names=['A', 'B', 'C', 'S'],filled=True)
plt.show()