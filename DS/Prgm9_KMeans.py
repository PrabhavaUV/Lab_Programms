import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Data/Mall_Customers.csv")
print(df.columns)

le=LabelEncoder()
sc=StandardScaler()

df["Genre"]=le.fit_transform(df["Genre"])
X=df[["Age","Genre","Annual Income (k$)","Spending Score (1-100)"]]

X=sc.fit_transform(X)

iner=[]

for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=42,n_init=10)
    km.fit(X)
    iner.append(km.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1,11),iner,marker="x")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

km=KMeans(n_clusters=5,random_state=42)
cl=km.predict(X)

plt.figure(figsize=(10,6))
sns.scatterplot(x=df["Annual Income (k$)"],y=df["Spending Score (1-100)"], hue=cl, palette="viridis", s=100)
plt.scatter(km.cluster_centers_[:,2],km.cluster_centers_[:,3],s=200,c="red",marker="x", label="Centroids")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("KMeans Cluster")
plt.legend("Clusters")
plt.show()