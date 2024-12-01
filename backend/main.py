from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
from gliner import GLiNER
from models.inference_input import InferenceInput
from fastapi.middleware.cors import CORSMiddleware

print("------------------ Initializing model... ------------------")
# Initialize GLiNER with the base model
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
print("------------------ Model initialized. ------------------")

origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "*"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "PUT", "UPDATE", "HEAD", "OPTIONS"],
    allow_headers=["Access-Control-Allow-Headers", 'Content-Type', 'Authorization', 'Access-Control-Allow-Origin']
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/inference")
async def inference(input: InferenceInput):
    labels_split = input.labels.split(',')
    # Perform entity prediction
    entities = model.predict_entities(input.text, labels_split, threshold=input.threshold)

    return entities

if __name__ == "__main__":

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
