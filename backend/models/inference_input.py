from pydantic import BaseModel

class InferenceInput(BaseModel):
    text : str
    labels: str
    threshold: float