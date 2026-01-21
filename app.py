from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# -----------------------------
# Load model & tokenizer ONCE
# -----------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# Load TRAINED model & tokenizer
# -----------------------------
MODEL_PATH = "./emotion_model"   # folder created by train_model.py

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.to(device)
model.eval()

label_names = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval',
    'caring', 'confusion', 'curiosity', 'desire', 'disappointment',
    'disapproval', 'disgust', 'embarrassment', 'excitement',
    'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness',
    'optimism', 'pride', 'realization', 'relief', 'remorse',
    'sadness', 'surprise', 'neutral'
]

emotion_emojis = {
    "admiration": "🤩",
    "amusement": "😄",
    "anger": "😡",
    "annoyance": "😒",
    "approval": "👍",
    "caring": "🤗",
    "confusion": "😕",
    "curiosity": "🧐",
    "desire": "😍",
    "disappointment": "😞",
    "disapproval": "👎",
    "disgust": "🤢",
    "embarrassment": "😳",
    "excitement": "🤩",
    "fear": "😨",
    "gratitude": "🙏",
    "grief": "😢",
    "joy": "😊",
    "love": "❤️",
    "nervousness": "😬",
    "optimism": "🌈",
    "pride": "🦁",
    "realization": "💡",
    "relief": "😌",
    "remorse": "😔",
    "sadness": "😞",
    "surprise": "😲",
    "neutral": "😐"
}

# -----------------------------
# Prediction function
# -----------------------------
def predict_emotion(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0][predicted_class].item()

    emotion = label_names[predicted_class]
    emoji = emotion_emojis.get(emotion, "🙂")
    confidence = round(confidence * 100, 2)

    return emotion, emoji, confidence


# -----------------------------
# Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    emoji = None
    confidence = None

    if request.method == "POST":
        text = request.form["text"]
        emotion, emoji, confidence = predict_emotion(text)

    return render_template(
    "index.html",
    emotion=emotion,
    emoji=emoji,
    confidence=confidence
)

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
