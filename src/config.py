from pathlib import Path

#Raiz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[1]

#---Vocabularios---
VOCAB_PATH= PROJECT_ROOT / "vocabulario"

#---Modelos---
RUNS_PATH = PROJECT_ROOT /  "runs"
BEST_MODEL_PATH = PROJECT_ROOT/ "model" / "best.pth"

#---Media---

MEDIA_PATH = PROJECT_ROOT / "media"