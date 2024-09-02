from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Preloads
symptoms = pd.read_csv('data/symptoms.csv')
cures = pd.read_csv('data/cures.csv', index_col=0)
vectorizer = TfidfVectorizer()
symptom_vectors = vectorizer.fit_transform(list(symptoms['SYMPTOM'].to_dict().values()))

@app.route('/', methods=['POST'])
def find_disease():
    user_input = request.json.get('user_input').split(", ")
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

if __name__ == '__main__':
    app.run(debug=True)
