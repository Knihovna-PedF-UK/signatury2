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

pd.set_option('max_colwidth', 1000)

df = pd.read_csv("signatury.tsv", sep="\t", header=0)

# musíme odstranit NaN hodnoty, jinak nám bude CountVectorizer hlásit chybu
df = df.fillna("")

print(df.head())


# pokud se signatura vyskytuje jen jednou, dojde při vektorizaci k chybě. 
# každá hodnota se musí vyskytovat aspoň dvakrát
value_counts = df.signatura.value_counts()

# projdeme počty signatur a odstraníme ty, co se vyskytujou jen jednou
for signatura, count  in value_counts.items():
    if count == 1:
        to_remove = df[df['signatura'] == signatura].index
        df.drop(to_remove, inplace=True)


count_vec = CountVectorizer()
unknown_vec = CountVectorizer()

# tady vypíšeme knížky bez signatury
unknowns = df[df['signatura'] == 'Unknown']
df = df.drop(df[df['signatura'] == 'Unknown'].index)


# tfvectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5)

# spojíme název knihy a klíčový slova. ty použijeme dvakrát, aby měly větší váhu
text = df['text'] + " " +  df['keywords'] + " " + df['keywords']
unknown_text = unknowns['text'] + " " +  unknowns['keywords'] + " " + unknowns['keywords']

# bow = count_vec.fit_transform(df['text'])
bow = count_vec.fit_transform(text)
print(bow)

unknown_search = unknown_vec.fit_transform(unknown_text)
# tfvectorizer má menší přesnost, kupodivu
# bow = tfvectorizer.fit_transform(text)
# tohle brutálně zpomalí zpracování, třeba 10x
# a očividně to není třeba
# bow = np.array(bow.todense())


 
# X = pd.DataFrame(np.vstack([bow,keywords]).T, columns = ['text', 'keywords'])
y = df['signatura']

# X_train, X_test, kw_train, kw_test, y_train, y_test = train_test_split(bow, keywords, y, test_size=0.3, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(bow, y, test_size=0.05, stratify=y)

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

unknown_pred = model.predict(unknown_search)

print(unknown_pred)

# for i in range(0,20):
#     # row_no = y_test.at[i]
#     row_no = ""
#     orig_sig = y_test.iloc[i]
#     print(y_pred[i], row_no, orig_sig) 


