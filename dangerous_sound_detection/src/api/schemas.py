from typing import List

from pydantic import BaseModel, Field

class Event(BaseModel):
    label: str
    start_time: float
    end_time: float
    confidence: float

class Prediction(BaseModel):
    source: str
    confidence_threshold: float | None = None
    noise_floor_dbfs: float | None = None
    events: List[Event]
    activities: List[Event] = Field(default_factory=list)
