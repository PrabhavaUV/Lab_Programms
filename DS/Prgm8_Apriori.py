import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
df=pd.read_csv("DS/Data/market_basket.csv")
print(df.columns)

df.isna().sum()
df.fillna('', inplace=True)
df_dum = pd.get_dummies(df)
items = apriori(df_dum,min_support=0.01, use_colnames=True)
print(items)

rules=association_rules(items,metric="confidence",min_threshold=0.1)
rules.sort_values(['support','confidence'],ascending=False)
print(rules)