import pickle
import re
import spacy
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# Load models and encoders
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open('clf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))
CATEGORY_SKILLS = pickle.load(open("category_skills.pkl", "rb"))  # Load skills from train_model.py

def clean_resume(txt):
    txt = re.sub(r'http\S+|www\S+', ' ', txt)
    txt = re.sub(r'\S+@\S+', ' ', txt)
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\d+', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt).strip()
    return txt

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not Found"

def extract_skills(text):
    skills_found = set()
    text = text.lower()
    for category, skills in CATEGORY_SKILLS.items():
        for skill in skills:
            if skill in text:
                skills_found.add(skill)
    return list(skills_found)

def analyze_resume(resume_text):
    cleaned_text = clean_resume(resume_text)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()

    predicted_category = model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)[0]

    extracted_name = extract_name(resume_text)
    extracted_skills = extract_skills(resume_text)

    required_skills = CATEGORY_SKILLS.get(predicted_category_name, [])
    missing_skills = [skill for skill in required_skills if skill not in extracted_skills]

    if not missing_skills:
        feedback = "Great! You have all the necessary skills for this role."
    else:
        feedback = f"You may want to improve in: {', '.join(missing_skills)}."

    return {
        "Name": extracted_name,
        "Skills": extracted_skills,
        "Predicted Category": predicted_category_name,
        "Missing Skills": missing_skills if missing_skills else "None",
        "Feedback": feedback
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        resume_text = data.get("resume", "")

        if not resume_text:
            return jsonify({"error": "No resume text provided"}), 400

        result = analyze_resume(resume_text)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
