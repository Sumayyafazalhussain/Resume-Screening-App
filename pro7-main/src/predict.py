
import os, re, joblib
from src.utils.text_cleaning import clean_for_tfidf
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL = os.path.join(BASE, "..", "models", "model.pkl")
VECT = os.path.join(BASE, "..", "models", "tfidf.pkl")
LE = os.path.join(BASE, "..", "models", "label_encoder.pkl")

ROLE_SKILLS = {
    "Data Science": ["python","machine learning","pandas","numpy","sql","nlp","deep learning"],
    "Software Engineer": ["java","c++","python","git","docker","api","microservices"],
    "Web Designing": ["html","css","javascript","figma","ui","ux","bootstrap"],
    "HR": ["recruitment","onboarding","hr","talent","interview","payroll"],
    "Business Development": ["sales","crm","lead generation","negotiation","b2b"],
    "Health": ["patient","clinical","nursing","healthcare","treatment"]
}

def load_artifacts():
    model = joblib.load(MODEL)
    vect = joblib.load(VECT)
    le = joblib.load(LE)
    return model, vect, le

def match_skills(text: str, label: str):
    text = text.lower()
    skills = ROLE_SKILLS.get(label, [])
    found = []
    for s in skills:
        if re.search(r"\\b" + re.escape(s) + r"\\b", text):
            found.append(s)
    return found

def predict_resume(text: str):
    model, vect, le = load_artifacts()
    clean = clean_for_tfidf(text)
    X = vect.transform([clean])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        idx = proba.argmax()
        score = float(proba[idx])*100
    else:
        idx = model.predict(X)[0]
        score = 100.0
    label = le.inverse_transform([idx])[0]
    skills = match_skills(text, label)
    return label, round(score,2), skills
