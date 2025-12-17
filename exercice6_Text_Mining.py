#### Exercice 6 du chapitre Text Mining - étape 1 à 4 #####

import requests
import bs4
import re
import unicodedata
import nltk
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download("stopwords")

from sklearn.feature_extraction.text import CountVectorizer

from nltk.stem import PorterStemmer


url = [
    "https://www.themoviedb.org/movie/12444-harry-potter-and-the-deathly-hallows-part-1", #Le deuxuième lien de l'exercice 6a a été retiré car le synopsis n'était pas disponible
    "https://www.themoviedb.org/tv/84773-the-lord-of-the-rings-the-rings-of-power",
    "https://www.themoviedb.org/tv/250033-icons-unearthed-lord-of-the-rings"
] 

print("#" * 60, "texte brut des synopsis")
### Get the synopses ###

documents = []


for x in url:
    response = requests.get(x)
    content = response.text
    soup = bs4.BeautifulSoup(content, 'html.parser')

    for div in soup.find_all("div", class_="overview"):
        synopsis = div.find("p").text.strip()
        documents.append(synopsis)


print(documents)

processed_docs = []

for doc in documents:
    # Lowercase
    doc = doc.lower()

    # Ponctuation removal
    doc = re.sub(r'[^\w\s]', '', doc)

    # Accents removal
    doc = ''.join(
        c for c in unicodedata.normalize('NFD', doc)
        if unicodedata.category(c) != 'Mn'
    )

    # Tokenization
    tokens = word_tokenize(doc)

    # Stopwords removal
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]

    processed_docs.append(tokens)

print("#" * 60, "texte nettoyé et normalisé après tokenization")
for tokens in processed_docs:
    print(tokens)
print("Number of tokens in total:", sum(len(tokens) for tokens in processed_docs))
print("#" * 60)

### create a term-document matrix ###

clean_docs = [] 

for doc in documents:  #Attention: we use the original documents here because this loop does not require tokenization
    doc = doc.lower()
    doc = re.sub(r'[^\w\s]', '', doc)
    doc = ''.join(
        c for c in unicodedata.normalize('NFD', doc)
        if unicodedata.category(c) != 'Mn'
    )
    clean_docs.append(doc)


vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(clean_docs)

df = pd.DataFrame(
    X.toarray(),
    columns=vectorizer.get_feature_names_out()
)

print(df)
print("Number of dimensions in the term-document matrix:", X.shape[1])

'''
print(X.toarray())
print(vectorizer.get_feature_names_out())
print("number of tokens:", len(vectorizer.get_feature_names_out()))  
'''

### Stemming (Porter Stemmer) ###

ps = PorterStemmer()
stemmed_docs = []
for tokens in processed_docs:
    stemmed_tokens = [ps.stem(token) for token in tokens]
    stemmed_docs.append(stemmed_tokens)

print("#" * 60, "texte après stemming")
for tokens in stemmed_docs:
    print(tokens)

print("Number of tokens in total after stemming:", sum(len(tokens) for tokens in stemmed_docs))
print("#" * 60)

### TDM after stemming ###

stemmed_clean_docs = []
for tokens in stemmed_docs:
    stemmed_clean_docs.append(' '.join(tokens)) 
vectorizer_stemmed = CountVectorizer()
X_stemmed = vectorizer_stemmed.fit_transform(stemmed_clean_docs)
df_stemmed = pd.DataFrame(
    X_stemmed.toarray(),
    columns=vectorizer_stemmed.get_feature_names_out()
)
print(df_stemmed)
print("Number of dimensions in the term-document matrix after stemming:", X_stemmed.shape[1])

print("#" * 60)
