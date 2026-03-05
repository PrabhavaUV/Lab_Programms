import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data/Cancer_Data.csv')

print(df.columns)
print(df.info())

df.isna().sum()
df.drop('Unnamed: 32', axis=1, inplace=True)
df['radius_mean'] = df['radius_mean'].fillna(df['radius_mean'].mean())
df['texture_mean'] = df['texture_mean'].fillna(df['texture_mean'].mean())
df.dropna(inplace=True)

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.boxplot(df['texture_mean'])
plt.title('Box plot for texture_mean')
plt.ylabel('Texture Mean')
plt.subplot(1, 2, 2)
plt.boxplot(df['radius_mean'])
plt.title('Box plot for radius_mean')
plt.ylabel('Radius Mean')
plt.show()

Q1_texture = df['texture_mean'].quantile(0.25)
Q3_texture = df['texture_mean'].quantile(0.75)
IQR_texture = Q3_texture - Q1_texture
Q1_radius = df['radius_mean'].quantile(0.25)
Q3_radius = df['radius_mean'].quantile(0.75)
IQR_radius = Q3_radius - Q1_radius
lower_bound_texture = Q1_texture - 1.5 * IQR_texture
upper_bound_texture = Q3_texture + 1.5 * IQR_texture
lower_bound_radius = Q1_radius - 1.5 * IQR_radius
upper_bound_radius = Q3_radius + 1.5 * IQR_radius
df_cleaned = df[(df['texture_mean'] >= lower_bound_texture) & (df['texture_mean'] <= upper_bound_texture) &
(df['radius_mean'] >= lower_bound_radius) & (df['radius_mean'] <= upper_bound_radius)]
print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", df_cleaned.shape)

y_cleaned = df_cleaned['diagnosis']
X_cleaned = df_cleaned[['texture_mean','radius_mean']]

from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
y = le.fit_transform(y_cleaned)

sc = StandardScaler()
X = sc.fit_transform(X_cleaned)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
k_values = list(range(1, 30))
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracies.append(accuracy)
for k, accuracy in zip(k_values, accuracies):
    print(f'k={k}, Accuracy: {accuracy:.4f}')
best_k = k_values[accuracies.index(max(accuracies))]
print(f'Best k: {best_k} with accuracy: {max(accuracies):.4f}')

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, predictions)
print(cm)
print("Classification Report: \n",classification_report(y_test, predictions))

ConfusionMatrixDisplay(cm).plot()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df_cleaned.select_dtypes(exclude=['object']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()