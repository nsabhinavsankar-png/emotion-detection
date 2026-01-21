import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# -----------------------------
# 1. Load dataset
# -----------------------------
print("Loading dataset...")
dataset = load_dataset("go_emotions", "simplified")

label_names = dataset["train"].features["labels"].feature.names

def fix_labels(example):
    example["labels"] = example["labels"][0]
    return example

dataset = dataset.map(fix_labels)
# -----------------------------
# 3. Reduce dataset size (IMPORTANT)
# -----------------------------
train_data = dataset["train"].shuffle(seed=42).select(range(12000))
val_data = dataset["validation"].shuffle(seed=42).select(range(2000))

# -----------------------------
# 4. Tokenizer
# -----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_data(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# Tokenize
train_data = train_data.map(tokenize_data, batched=True)
val_data = val_data.map(tokenize_data, batched=True)

# Remove unused columns
columns_to_remove = train_data.column_names
columns_to_remove.remove("labels")
columns_to_remove.remove("input_ids")
columns_to_remove.remove("attention_mask")

train_data = train_data.remove_columns(columns_to_remove)
val_data = val_data.remove_columns(columns_to_remove)

# Set format for PyTorch
train_data.set_format("torch")
val_data.set_format("torch")

# -----------------------------
# 5. Model (SINGLE-LABEL)
# -----------------------------
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels= 28,
    problem_type="single_label_classification"
)
model.to(device)
model.eval()

# -----------------------------
# 6. Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=5,
    learning_rate=2e-5,
    logging_dir="./logs",
    report_to="none"
)

# -----------------------------
# 7. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer
)

# -----------------------------
# 8. Train
# -----------------------------
print("Starting training...")
trainer.train()

print("Saving trained model...")
model.save_pretrained("./emotion_model")
tokenizer.save_pretrained("./emotion_model")

# -----------------------------
# 9. Test predictions
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

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return label_names[predicted_class]

trainer.save_model("emotion_model")
tokenizer.save_pretrained("emotion_model")

print("\nTEST PREDICTIONS")
print("Joy:", predict_emotion("I am very happy today"))
print("Anger:", predict_emotion("This is the worst experience ever"))
print("Fear:", predict_emotion("I feel scared and nervous"))