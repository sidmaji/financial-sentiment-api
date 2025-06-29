from transformers import BertForSequenceClassification, BertTokenizer, pipeline

model_path = "finbert"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)


def classify(text: str):
    result = pipe(text)[0]
    return {"label": result["label"], "score": round(result["score"] * 100, 2)}
