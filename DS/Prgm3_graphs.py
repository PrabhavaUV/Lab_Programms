import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("Data/mpg_raw.csv")

# Convert horsepower to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')

# Histogram (Continuous Variables)
plt.figure(figsize=(10,6))
df[['mpg', 'displacement', 'horsepower', 'acceleration']].hist(figsize=(10,8))
plt.suptitle("Histogram of Continuous Variables")
plt.show()

# Scatterplot (Relationship between two continuous variables)
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='weight', y='mpg')
plt.title("Scatterplot: Weight vs MPG")
plt.show()

# Countplot (Categorical Variable)
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='cylinders')
plt.title("Countplot of Cylinders")
plt.show()

# Pointplot
plt.figure(figsize=(8,5))
sns.pointplot(data=df, x='model_year', y='mpg')
plt.title("Pointplot: Model Year vs MPG")
plt.show()