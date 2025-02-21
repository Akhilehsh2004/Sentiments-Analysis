from flask import Flask, request, jsonify, render_template
import pickle
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from flair.models import TextClassifier
from flair.data import Sentence

app = Flask(__name__)

# Load sentiment analyzers
sia = SentimentIntensityAnalyzer()
flair_classifier = None
MODEL_PATH = "sentiment_model.pkl"

def get_flair_classifier():
    global flair_classifier
    if flair_classifier is None:
        try:
            flair_classifier = TextClassifier.load('sentiment')
        except Exception as e:
            print(f"Error loading Flair classifier: {e}")
            return None
    return flair_classifier

# Load pre-trained model if available
def load_model():
    global flair_classifier
    try:
        with open(MODEL_PATH, 'rb') as f:
            flair_classifier = pickle.load(f)
        print("✅ Model loaded successfully.")
    except FileNotFoundError:
        print("⚠️ Model file not found. Loading a new model.")
        get_flair_classifier()
    except Exception as e:
        print(f"❌ Error loading model: {e}")

load_model()

# Clean text function
def clean_text(text):
    return re.sub(r'[^a-zA-Z\s]', '', text).lower()

# Sentiment Analysis Function
def analyze_sentiment(text):
    try:
        cleaned_text = clean_text(text)
        vader_score = sia.polarity_scores(cleaned_text)['compound']
        blob_score = TextBlob(cleaned_text).sentiment.polarity

        flair_classifier = get_flair_classifier()
        flair_score, flair_label = 0, "NEUTRAL"
        if flair_classifier:
            sentence = Sentence(text)
            flair_classifier.predict(sentence)
            flair_sentiment = sentence.labels[0].to_dict()
            flair_score = flair_sentiment['confidence']
            flair_label = flair_sentiment['value']

        final_score = (0.25 * vader_score + 0.25 * blob_score + 0.50 * (flair_score if flair_label == 'POSITIVE' else -flair_score))
        sentiment = "Positive 😊" if final_score > 0.05 else "Negative 😠" if final_score < -0.05 else "Neutral 😐"

        return {"text": text, "sentiment": sentiment, "final_score": final_score, "vader_score": vader_score, "blob_score": blob_score, "flair_score": flair_score}
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {"error": "Sentiment analysis failed"}

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    result = analyze_sentiment(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
