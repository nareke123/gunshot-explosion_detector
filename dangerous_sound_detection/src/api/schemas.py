from pydantic import BaseModel
from typing import List

class Event(BaseModel):
    label: str
    start_time: float
    end_time: float
    confidence: float

class Prediction(BaseModel):
    source: str
    events: List[Event]