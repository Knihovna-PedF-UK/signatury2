import pandas as pd
import numpy as np
from sys import argv
import string


import marcpandas 
import almaxml 

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


def load_data(marcfile, almafile):
    # load XML files into pandas
    a = marcpandas.load(marcfile)
    b = almaxml.load(almafile)
    # join loaded files
    d = pd.concat([a,b], axis=1)
    # remove records without titles
    d = d[d['title'].notna()]
    return d

# přidáme prefix kw_ ke všem klíčovým slovům, trošku to zlepší přesnost
def add_kw(x):
    temp = []
    for word in x.split(" "):
        temp.append("kw_" + word)
    y =  " ".join(temp)
    return y

def clean_data(df):
    # pokud se signatura vyskytuje jen jednou, dojde při vektorizaci k chybě. 
    # každá hodnota se musí vyskytovat aspoň dvakrát
    value_counts = df.signatura.value_counts()

    # projdeme počty signatur a odstraníme ty, co se vyskytujou jen jednou
    for signatura, count  in value_counts.items():
        if count == 1:
            to_remove = df[df['signatura'] == signatura].index
            df.drop(to_remove, inplace=True)
    
    # přidáme prefix kw_ ke všem klíčovým slovům
    # df['keywords'] = df['keywords'].apply(add_kw)
    df['keywords'] = [add_kw(kw) for kw in df['keywords']]

    # spojíme název knihy a klíčový slova. ty použijeme dvakrát, aby měly větší váhu
    df['search'] = df['title'] + " " +  df['keywords'] + " " + df['keywords']
    return df

def split_data(df):
    # získáme dva soubory, u jedněch signaturu známe, u druhých ne
    unknown = df[df['signatura'] == "Unknown"]
    known   = df[df['signatura'] != "Unknown"]
    return known, unknown

def search(known, unknown):
    # tahle funkce obsahuje spoustu kódu z dřívějších pokusů
    # tfvectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5)
    
    known_text = known['search']
    unknown_text = unknown['search']
    # unknown_text = unknown['text'] + " " + unknown['keywords'] + " " + unknown['keywords']
    
    # nejdřív zvektorizujeme všechny tokeny. používáme bigramy, podstatně to zvyšuje přesnost
    count_vec = CountVectorizer(analyzer='word',ngram_range=(2,2), max_df=0.8)
    count_vec.fit(df['search'])
    # bow = count_vec.fit_transform(df['text'])
    # učící data
    bow = count_vec.transform(known_text)
    # knížky bez signatury k vyhledání
    to_find = count_vec.transform(unknown_text)
    # print(bow)
    
    # tfvectorizer má menší přesnost, kupodivu
    # bow = tfvectorizer.fit_transform(text)
    # tohle brutálně zpomalí zpracování, třeba 10x
    # a očividně to není třeba
    # bow = np.array(bow.todense())
     
    # X = pd.DataFrame(np.vstack([bow,keywords]).T, columns = ['text', 'keywords'])
    y = known['signatura']
    
    # tohle jsou testovací data,už nepotřebujeme
    # X_train, X_test, kw_train, kw_test, y_train, y_test = train_test_split(bow, keywords, y, test_size=0.3, stratify=y)
    # X_train, X_test, y_train, y_test = train_test_split(bow, y, test_size=0.25, stratify=y)
    
    # tohle je brutálně pomalý a vypisuje chyby, očividně to potřebuje ty data ještě upravit
    # model = LogisticRegressionCV().fit(X_train, y_train)
    # --------------
    # Naive Bayes - nejrychlejší a zároveň nejpřesnější
    # model = MultinomialNB().fit(X_train, y_train)
    model = MultinomialNB().fit(bow, y)
    # ------------------
    # o trošku pomalejší, než Bayes, přesnost podobná
    # model = svm.SVC().fit(X_train, y_train)
    # ------------------
    # smodel = SelfTrainingClassifier(SGDClassifier())
    # model = smodel.fit(X_train, y_train)
    # --------------
    # y_pred = model.predict(X_test)
    y_pred = model.predict(to_find)
    
    # vyhodnocení testovacích dat
    # print('Accuracy:', accuracy_score(y_test, y_pred))
    # print('F1 score:', f1_score(y_test, y_pred, average="macro"))
    
    # print(y_pred)
    # print(y_test)
    return y_pred

def make_shortcut(title, author):
    # ostraníme mezery a nechtěné znaky z názvu
    clean_title = "".join(title.split(" ")).strip(string.punctuation)
    # zkonstruujeme zkrácený kód z autora a názvu knížky
    short_author = ""
    short_title  = clean_title[0:2]
    # pokud autor chybí, použijeme 4 písmena z názvu
    if pd.notna(author):
        short_author = author[0:2]
    else:
        short_title = clean_title[0:4]
    return f"{short_author}{short_title}"

def save(filename, unknown, y_pred):
    unknown.insert(2, "pred", y_pred)
    unknown.to_excel(odsname, engine="odf", index = False)

script, marcfile, almafile = argv

# df = pd.read_csv("signatury.tsv", sep="\t", header=0)

df = load_data(marcfile, almafile)
df = clean_data(df)
known, unknown = split_data(df)
y_pred = search(known, unknown)

# vytvoříme zkratku pro řazení na regále
unknown["short"]  = [make_shortcut(title, author) for title, author in zip(unknown["title"], unknown["author"])]

i = 0
# print(unknown.head(20))
for id in y_pred:
    curr = unknown.iloc[i]
    print(y_pred[i], curr['id'], curr['title'], curr['keywords'])
    # print(y_pred[i], rec.at[id, 'text'], rec.at[id, 'keywords'])
    i=i+1

odsname = "signatury.ods"
print(f"Writing to: {odsname}")
save(odsname, unknown, y_pred)

# print(curr.iloc[y_pred])

# for id, signatura in df[df['signatura']=="Unknown"].items():
    # print(id,X_test[id])


