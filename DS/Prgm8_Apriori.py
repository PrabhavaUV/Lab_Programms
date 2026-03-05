import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
df=pd.read_csv("Data/Market_Basket_Optimisation.csv")
print(df.columns)
df.isna().sum()
df.fillna(inplace=True)
item=apriori(df,min_support=0.01,use_colnames=True)
print(item)
rules=association_rules(item,metric="confidence",min_threshold=0.1)
rules.sort(['support','confidence'],ascending=False)
print(rules)