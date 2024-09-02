from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Flask, request, jsonify
import json
import firebase_admin
from firebase_admin import credentials, storage

# Initialize the Firebase Admin SDK
cred = credentials.Certificate('ai-nurse-f5508-firebase-adminsdk-nggiz-826f82fb35.json')
firebase_admin.initialize_app(cred, {'storageBucket': 'ai-nurse-f5508.appspot.com'})

app = Flask(__name__)

# Preloads
symptoms = pd.read_csv('data/symptoms.csv')
cures = pd.read_csv('data/cures.csv', index_col=0)
vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(list(symptoms['SYMPTOM'].to_dict().values()))

@app.route('/get', methods=['GET'])
def get():
    if request.args.get("id") == "rem": return all_rem(request.args.get("user_id"))

def all_rem(folder_name):
    bucket = storage.bucket()
    blobs = bucket.list_blobs(prefix=f'{folder_name}/')

    data = {}
    for blob in blobs:
        if blob.name.endswith('.json'):
            try:
                content = blob.download_as_text()
                data = json.loads(content)
            except Exception as e:
                print(f'Error reading file {blob.name}: {e}')
    
    return data

@app.route('/post', methods=['POST'])
def post():
    if request.json.get('id') == 'disease': return find_disease(request.json.get('user_input'))
    elif request.json.get('id') == 'set': return set_reminder(request.json)

def find_disease(user_input):
    return_obj = {'disease': '', 'dietary_recommendations': '', 'medicine': ''}

    def find_closest_symptoms(user_text: str):
        # Function to find the most similar term using scikit-learn
        global vectorizer, symptom_vectors
        user_vector = vectorizer.transform([user_text])
        
        # Compute cosine similarity
        similarity_scores = cosine_similarity(user_vector, symptom_vectors).flatten()
        
        # Find the index of the most similar term
        top_3_indices = similarity_scores.argsort()[::-1][:3]
        
        # Get the top n terms and their corresponding similarity scores
        top_3_terms = {}
        for i in top_3_indices:
            if similarity_scores[i] > 0.4:
                top_3_terms[i] = similarity_scores[i]
        
        return top_3_terms
    
    user_symptom = {}
    for i in user_input:
        x = find_closest_symptoms(i)
        for j in x:
            try: user_symptom[j] += x[j]
            except KeyError: user_symptom[j] = x[j]
    
    diseases = {}
    for i in user_symptom:
        for j in eval(symptoms.loc[i, "DISEASES"]):
            try: diseases[j] += user_symptom[i]
            except KeyError: diseases[j] = user_symptom[i]

    return_obj['disease'] = max(diseases, key=lambda x: diseases[x])
    return_obj['dietary_recommendations'] = eval(cures.loc[return_obj['disease'], 'dietary_recommendations'])
    return_obj['medicine'] = eval(cures.loc[return_obj['disease'], 'medicine'])
    print(return_obj)

    return jsonify(return_obj)

def set_reminder(data):
    # Save JSON string to a file
    with open('data.json', 'w') as json_file:
        json.dump(data, json_file)

    # Reference the storage bucket
    bucket = storage.bucket()

    # Reference to the file in Firebase Storage
    file_path = f'{data['user_id']}/reminder.json'
    blob = bucket.blob(file_path)

    # Download the existing file into memory
    existing_data = {}
    if blob.exists():
        existing_data = json.loads(blob.download_as_text())
    
    # New data to append
    new_data = {
        f"{len(existing_data)}": data
    }

    # Append the new data to existing data
    existing_data.update(new_data)

    # Convert updated data to JSON string
    updated_json_data = json.dumps(existing_data)

    # Upload the updated file
    blob.upload_from_string(updated_json_data, content_type='application/json')

    return {"status":"ok"}

if __name__ == '__main__':
    app.run(debug=True)
