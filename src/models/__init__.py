"""Modelos con interfaz común fit/predict y un registry para el config."""
from src.models.base import BaseModel  
from src.models.dummy import MeanPredictor  
from src.models.registry import REGISTRY, build_model  
