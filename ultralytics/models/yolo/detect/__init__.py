# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor, YOLOv10DetectionPredictor
from .train import DetectionTrainer, YOLOv10DetectionTrainer
from .val import DetectionValidator, YOLOv10DetectionValidator

__all__ = ("DetectionPredictor", "DetectionTrainer", "DetectionValidator",
           "YOLOv10DetectionPredictor", "YOLOv10DetectionTrainer", "YOLOv10DetectionValidator")
