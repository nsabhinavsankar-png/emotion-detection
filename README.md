# Emotion Detection from Text 

A deep learning web application that detects human emotions from English text using a fine-tuned DistilBERT model.

## Features
- Detects 28 emotions
- Shows confidence score
- Emoji representation of emotion
- Clean dark-themed UI
- Flask backend

## Screenshots

### Basic UI
![Basic UI](screenshots/basic_ui.png)

### Working UI (Emotion Prediction)
![Working UI](screenshots/working_ui.png)

## Model
- DistilBERT (Hugging Face)
- GoEmotions Dataset

## Tech Stack
- Python
- PyTorch
- Hugging Face Transformers
- Flask
- HTML & CSS

## How to Run

```bash
# activate venv
source venv/bin/activate

# run app
python app.py