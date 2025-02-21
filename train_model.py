import numpy as np
import pandas as pd
import re
import pickle
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

nlp = spacy.load("en_core_web_sm")

df = pd.read_csv('UpdatedResumeDataSet.csv')

CATEGORY_SKILLS = {
    "Data Science": ["machine learning", "deep learning", "python", "data analysis", "sql"],
    "Software Engineer": ["java", "c++", "python", "cloud computing", "docker"],
    "Artificial Intelligence": ["machine learning", "deep learning", "nlp", "computer vision"],
    "Web Development": ["html", "css", "javascript", "react", "node.js"],
}

def clean_resume(txt):
    txt = re.sub(r'http\S+|www\S+', ' ', txt)
    txt = re.sub(r'\S+@\S+', ' ', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\d+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

df['Resume'] = df['Resume'].apply(clean_resume)

le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])
pickle.dump(le, open("encoder.pkl", 'wb'))

tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['Resume'])
y = df['Category']
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.toarray()
X_test = X_test.toarray()

models = {
    "KNN": OneVsRestClassifier(KNeighborsClassifier()),
    "SVC": OneVsRestClassifier(SVC()),
    "RandomForest": OneVsRestClassifier(RandomForestClassifier())
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

pickle.dump(best_model, open('clf.pkl', 'wb'))
pickle.dump(CATEGORY_SKILLS, open("category_skills.pkl", "wb"))
print(f"Best Model Saved: {best_model}")
