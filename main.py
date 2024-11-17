import model
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"hello": "world"}


@app.get("/test")
async def test_news(text: str):
    print(text)
    pred = model.test_article(text)

    return {"prediction": pred}


@app.post("/train")
async def train_model(text: str):
    accuracy=model.train_model(text)
    return {"accuracy": accuracy}
