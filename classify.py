import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib 
import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('max_colwidth', 1000)

df = pd.read_csv("signatury.tsv", sep="\t", header=0)

# musíme odstranit NaN hodnoty, jinak nám bude CountVectorizer hlásit chybu
df = df.fillna("")

# print(df.head())


# pokud se signatura vyskytuje jen jednou, dojde při vektorizaci k chybě. 
# každá hodnota se musí vyskytovat aspoň dvakrát
value_counts = df.signatura.value_counts()

# projdeme počty signatur a odstraníme ty, co se vyskytujou jen jednou
for signatura, count  in value_counts.items():
    if count == 1:
        to_remove = df[df['signatura'] == signatura].index
        df.drop(to_remove, inplace=True)



# přidáme prefix kw_ ke všem klíčovým slovům, trošku to zlepší přesnost
def add_kw(x):
    temp = []
    for word in x.split(" "):
        temp.append("kw_" + word)
    y =  " ".join(temp)
    return y

df['keywords'] = df['keywords'].apply(add_kw)






# tfvectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5)

# spojíme název knihy a klíčový slova. ty použijeme dvakrát, aby měly větší váhu
text = df['text'] + " " +  df['keywords'] + " " + df['keywords']
unknown = df[df['signatura'] == "Unknown"]

unknown_text = unknown['text'] + " " + unknown['keywords'] + " " + unknown['keywords']

count_vec = CountVectorizer(analyzer='word',ngram_range=(2,2), max_df=0.8)
# bow = count_vec.fit_transform(df['text'])
bow = count_vec.fit_transform(text)
# print(bow)

# tfvectorizer má menší přesnost, kupodivu
# bow = tfvectorizer.fit_transform(text)
# tohle brutálně zpomalí zpracování, třeba 10x
# a očividně to není třeba
# bow = np.array(bow.todense())


 
# X = pd.DataFrame(np.vstack([bow,keywords]).T, columns = ['text', 'keywords'])
y = df['signatura']

def filter_unknown(X,y):
    pass



# X_train, X_test, kw_train, kw_test, y_train, y_test = train_test_split(bow, keywords, y, test_size=0.3, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(bow, y, test_size=0.25, stratify=y)

# tohle je brutálně pomalý a vypisuje chyby, očividně to potřebuje ty data ještě upravit
# model = LogisticRegressionCV().fit(X_train, y_train)
# --------------
# Naive Bayes
model = MultinomialNB().fit(X_train, y_train)
# ------------------
# o trošku pomalejší, než Bayes, přesnost podobná
# model = svm.SVC().fit(X_train, y_train)
# ------------------
# smodel = SelfTrainingClassifier(SGDClassifier())
# model = smodel.fit(X_train, y_train)
# --------------
y_pred = model.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred, average="macro"))

# print(y_pred)
# print(y_test)



i = 0
for id, signatura in y_test.head(20).items():
    print(y_pred[i], signatura, df.at[id, 'text'], df.at[id, 'keywords'])
    i=i+1

# for id, signatura in df[df['signatura']=="Unknown"].items():
    # print(id,X_test[id])


