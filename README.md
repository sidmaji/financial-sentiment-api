# 🧠 Financial Sentiment API

A FastAPI-based backend for multi-model financial sentiment analysis. Supports:

-   FinBERT (transformer-based)
-   Azure o4-mini (zero-shot LLM)
-   Logistic Regression (TF-IDF)

## 🚀 Endpoints

### POST `/logistic-regression`

```json
{ "text": "Tesla beats Q2 earnings expectations" }
```
