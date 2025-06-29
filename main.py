from fastapi import FastAPI
from pydantic import BaseModel

# from models import azure_o4, finbert, logistic_regression
from models import logistic_regression

app = FastAPI()


class SentimentRequest(BaseModel):
    text: str


@app.post("/logreg")
def logreg(request: SentimentRequest):
    return logistic_regression.classify(request.text)


# @app.post("/finbert")
# def finbert_api(request: SentimentRequest):
#     return finbert.classify(request.text)


# @app.post("/o4mini")
# def o4_api(request: SentimentRequest):
#     return azure_o4.classify(request.text)
