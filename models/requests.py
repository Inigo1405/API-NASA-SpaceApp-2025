from typing import Dict
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request model for /predict endpoint.

    Contains a mapping of feature names to numeric values that the model
    will consume. Keep it minimal so imports from `models.requests`
    succeed when the app starts.
    """
    features: Dict[str, float]