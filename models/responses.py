from typing import Dict, List, Optional
from pydantic import BaseModel

class PredictResponse(BaseModel):
    label: str
    probabilities: Optional[Dict[str, float]] = None
    recognized: List[str]
    unknown: List[str]
    missing: List[str]
    feature_order: List[str]
    engineered: Dict[str, Optional[float]]
