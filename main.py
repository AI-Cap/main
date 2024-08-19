from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify

app = Flask(__name__)

# Preloads
cred = credentials.Certificate("ai-nurse-f5508-firebase-adminsdk-nggiz-0e392a2b93.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

SYMPTOMS = [
    "Short Stature", "Short Limbs", "Spinal Stenosis", "Frequent Ear Infections", "Pimples", "Blackheads", "Whiteheads", "Possible Scarring on the Face, Chest, and Back", "Fatigue", "Weight Loss", "Low Blood Pressure", "Darkened Skin", "Swelling", "Organ Failure", "Weakness", "Pale or Yellowish Skin", "Irregular Heartbeats", "Shortness of Breath", "Dizziness", "Joint Pain", "Stiffness", "Decreased Range of Motion", "Chest Tightness", "Wheezing", "Coughing", "Mouth Sores", "Genital Sores", "Eye Inflammation", "Skin Lesions", "Lump in the Breast", "Changes in Breast Shape", "Skin Dimpling", "Nipple Discharge", "Persistent Cough", "Production of Mucus", "Chest Discomfort", "Abdominal Pain", "Bloating", "Diarrhea", "Constipation", "Fever", "Swelling at Infection Site", "Heart and Digestive Problems", "Itchy Rash that Turns into Fluid-Filled Blisters", "Tiredness", "Headache", "Watery Diarrhea", "Vomiting", "Rapid Dehydration", "Muscle Cramps", "Changes in Bowel Habits", "Blood in Stool", "Abdominal Discomfort", "Unexplained Weight Loss", "Sneezing", "Runny Nose", "Sore Throat", "Mild Fever", "Lung Infections", "Digestive Issues", "High Fever", "Severe Headache", "Pain Behind the Eyes", "Joint and Muscle Pain", "Rash", "Bleeding Gums", "Low Platelet Count", "Persistent Sadness", "Loss of Interest", "Changes in Appetite or Weight", "Sleep Disturbances", "Difficulty Concentrating", "Feelings of Worthlessness", "Increased Thirst", "Frequent Urination", "Extreme Hunger", "Unintended Weight Loss", "Blurred Vision", "Muscle Weakness", "Difficulty Walking", "Respiratory Problems", "Fragile Skin", "Blisters", "Skin Infections", "Scarring", "Stomach Pain", "Cramping", "Enlarged Spleen and Liver", "Bone Pain", "Painful Urination", "Abnormal Discharge from the Genitals", "Pelvic Pain or Bleeding Between Periods", "Severe Joint Pain (Often in the Big Toe)", "Redness", "Heat", "Nausea", "Jaundice", "Kidney Stones", "Depression", "No Symptoms Until a Fracture Occurs", "Back Pain", "Loss of Height", "Stooped Posture", "Intellectual Disability", "Behavioral Problems", "Seizures", "High Blood Pressure", "Headaches", "Sweating", "Rapid Heartbeat", "Cough", "Phlegm Production", "Red Patches of Skin Covered with Thick, Silvery Scales", "Dry and Cracked Skin", "Itching", "Burning", "Soreness", "Inflamed Eyes", "Rash Starting on the Face", "Swollen Lymph Nodes", "Gum Disease", "Bruising", "Anemia Due to Vitamin C Deficiency", "New or Changing Mole", "Asymmetry in Moles", "Irregular Borders", "Varied Colors in a Mole", "Painless Sore (Chancre)", "Skin Rashes", "Mucous Membrane Lesions", "Late-Stage Symptoms Affecting Organs like the Heart and Brain", "Muscle Stiffness", "Jaw Locking (Trismus)", "Difficulty Swallowing", "Muscle Spasms", "Weak Bones", "Severe Pain in the Side or Back", "Blood in Urine", "Chest Pain", "Coughing Up Blood", "Chills"
]
vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(SYMPTOMS)

@app.route('/find_disease', methods=['POST'])
def find_disease():
    user_input = request.json.get('user_input').split(", ")
    console.log('a')
    return_obj = {'disease': '', 'cures': '', 'dietary_recommendations': '', 'medicine': ''}

    def read_document(collection_name: str, document_id: str):
        doc_ref = db.collection(collection_name).document(document_id)
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()
        else:
            raise ValueError(f"Document not found with id '{document_id}' in collection '{collection_name}'")

    def find_closest_symptoms(user_text: str):
        # Function to find the most similar term using scikit-learn
        global vectorizer, SYMPTOMS, symptom_vectors
        user_vector = vectorizer.transform([user_text])
        
        # Compute cosine similarity
        similarity_scores = cosine_similarity(user_vector, symptom_vectors).flatten()
        
        # Find the index of the most similar term
        top_3_indices = similarity_scores.argsort()[::-1][:3]
        
        # Get the top n terms and their corresponding similarity scores
        top_3_terms = {}
        for i in top_3_indices:
            if similarity_scores[i] > 0.4:
                top_3_terms[SYMPTOMS[i]] = similarity_scores[i]
        
        return top_3_terms
    
    user_symptom = {}
    for i in user_input:
        x = find_closest_symptoms(i)
        for j in x:
            try: user_symptom[j] += x[j]
            except KeyError: user_symptom[j] = x[j]
    console.log('b')
    diseases = {}
    for i in user_symptom:
        for j in read_document("symptoms", i)["diseases"]:
            try: diseases[j] += user_symptom[i]
            except KeyError: diseases[j] = user_symptom[i]
    console.log('c')
    return_obj['disease'] = max(diseases, key=lambda x: diseases[x])
    return_obj.update(read_document("cures", return_obj['disease']))

    return jsonify(return_obj)

if __name__ == '__main__':
    app.run(debug=True)
