from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import pickle
import numpy as np
import requests

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}})  # Limit to specific origins as needed

# Load the BioBERT model with increased max length
model = pipeline("question-answering", model="dmis-lab/biobert-v1.1", max_length=512, truncation=True)

# Load the pre-trained ML model
with open('random_forest_model.pkl', 'rb') as f:
    ml_model = pickle.load(f)

# SerpApi credentials
SERPAPI_KEY = 'f79edf76b94ff7a7d8d7f90c5b7f665a2f5d9e8534937cc54040702c06231b6c'  # Replace with your actual SerpApi key

def get_context_from_serpapi(query):
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'organic_results' in data and len(data['organic_results']) > 0:
            # Concatenate multiple snippets to provide a more comprehensive context
            context = ""
            additional_references = []
            for result in data['organic_results'][:3]:
                if len(context) + len(result['snippet']) < 512:  # Limit the context length
                    context += result['snippet'] + " "
                    additional_references.append(result['link'])
                else:
                    break
            reference = data['organic_results'][0]['link']
            return context.strip(), reference, additional_references
    return None, None, None

def generate_fallback_response(query):
    # Provide a default response if the generated answer is not satisfactory
    return "I'm sorry, I couldn't find a relevant answer to your question. Please try rephrasing it or providing more details.", None, None

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()

    # Debugging lines to confirm received data and structure
    print("Received data:", data)

    if 'message' not in data:
        return jsonify({'error': 'Missing "message" in request.'}), 400

    prompt = data['message']
    context, reference, additional_references = data.get('context', ""), data.get('reference', ""), data.get('additional_references', [])  # Set default context and reference as empty if not provided

    if not context:
        context, reference, additional_references = get_context_from_serpapi(prompt)
        if not context:
            return jsonify({'error': 'Could not find relevant context for the question.'}), 404

    try:
        result = model(question=prompt, context=context)
        if not result['answer'] or len(result['answer']) < 200:
            # If the answer is too short or not satisfactory, use the fallback response
            answer, reference, additional_references = generate_fallback_response(prompt)
        else:
            answer = result['answer']

        # Concatenate the context and the answer
        response = f"{context}\n\n{answer}"
        return jsonify({'response': response, 'reference': reference, 'additional_references': additional_references})
    except Exception as e:
        # Catch any errors in model processing and provide error message
        print("Error during model processing:", e)
        return jsonify({'error': 'Processing error occurred.'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Debugging lines to confirm received data and structure
    print("Received data:", data)

    if 'features' not in data:
        return jsonify({'error': 'Missing "features" in request.'}), 400

    features = data['features']
    if not isinstance(features, list) or len(features) != 5:
        return jsonify({'error': 'Features must be a list of 5 numerical values.'}), 400

    try:
        # Convert features to numpy array
        features = np.array(features).reshape(1, -1)
        prediction = ml_model.predict(features)
        probability = ml_model.predict_proba(features)
        return jsonify({'prediction': int(prediction[0]), 'probability': probability.tolist()})
    except Exception as e:
        # Catch any errors in model processing and provide error message
        print("Error during model processing:", e)
        return jsonify({'error': 'Processing error occurred.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
