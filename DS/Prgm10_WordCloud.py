import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS
from nltk.probability import FreqDist
from PIL import Image
import numpy as np

df=open("Data/Word_Cloud.txt","r").read()
words=word_tokenize(df)
words = [word.lower() for word in words if word.isalpha()]
words = [word for word in words if word not in stopwords.words("english")]

st=PorterStemmer()
lem=WordNetLemmatizer()

stemed=[st.stem(word) for word in words]
lemed=[lem.lemmatize(word) for word in stemed]
processed=" ".join(lemed)

wordtk=word_tokenize(processed)
fd=FreqDist(wordtk)
top=fd.most_common(40)
print(top)

wc=WordCloud(width=500,height=500,background_color="white",random_state=42,collocations=False,stopwords=STOPWORDS).generate_from_frequencies(dict(top))

plt.figure(figsize=(10,6))
plt.imshow(wc)
plt.axis("off")
plt.title("Word Cloud of Most Common Words")
plt.show()

mask=np.array(Image.open("Data/Ghost.jpg"))
mask=255-mask
wc=WordCloud(width=500,height=500,background_color="white",random_state=42,collocations=False,stopwords=STOPWORDS,mask=mask).generate_from_frequencies(dict(top))
plt.figure(figsize=(10,6))
plt.imshow(wc)
plt.axis("off")
plt.title("Custom Shape Word Cloud")
plt.show()
