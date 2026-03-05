import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("Data/mpg_raw.csv")

# Structure & summary
print(df.info())
print(df.describe())

# Handle missing values
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)

# Histogram
sns.histplot(df['mpg'], kde=True); plt.show()

# Violin plot
sns.violinplot(x=df['cylinders']); plt.show()

# Boxplot (outliers)
sns.boxplot(x=df['horsepower']); plt.show()

# Heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm"); plt.show()

# Standardization
cols = ['mpg','displacement','horsepower','weight','acceleration']
df[cols] = StandardScaler().fit_transform(df[cols])