import os
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(message)s"
)

project_name = "music_recommendation"

list_of_files = [

    # ==========================
    # CONFIGS
    # ==========================
    "configs/config.yaml",
    "configs/model.yaml",
    "configs/database.yaml",

    # ==========================
    # DATA
    # ==========================
    "data/raw/.gitkeep",
    "data/interim/.gitkeep",
    "data/processed/.gitkeep",

    # ==========================
    # NOTEBOOKS
    # ==========================
    "notebooks/.gitkeep",

    # ==========================
    # SOURCE
    # ==========================
    f"src/{project_name}/__init__.py",

    # Pipeline
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/data_ingestion.py",
    f"src/{project_name}/pipeline/preprocessing.py",
    f"src/{project_name}/pipeline/feature_engineering.py",
    f"src/{project_name}/pipeline/content_training.py",
    f"src/{project_name}/pipeline/als_training.py",
    f"src/{project_name}/pipeline/hybrid_training.py",
    f"src/{project_name}/pipeline/evaluation.py",

    # Recommender
    f"src/{project_name}/recommender/__init__.py",
    f"src/{project_name}/recommender/content.py",
    f"src/{project_name}/recommender/collaborative.py",
    f"src/{project_name}/recommender/hybrid.py",
    f"src/{project_name}/recommender/gnn.py",

    # Vector Store
    f"src/{project_name}/vector_store/__init__.py",
    f"src/{project_name}/vector_store/faiss_store.py",

    # Database
    f"src/{project_name}/database/__init__.py",
    f"src/{project_name}/database/database.py",

    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/utils/logger.py",

    # ==========================
    # MODELS
    # ==========================
    "models/.gitkeep",

    # ==========================
    # ARTIFACTS
    # ==========================
    "artifacts/.gitkeep",

    # ==========================
    # MLFLOW
    # ==========================
    "mlruns/.gitkeep",

    # ==========================
    # BACKEND (FastAPI)
    # ==========================
    "backend/app/__init__.py",
    "backend/app/main.py",

    "backend/app/api/__init__.py",
    "backend/app/api/recommend.py",
    "backend/app/api/search.py",
    "backend/app/api/song.py",

    "backend/app/services/__init__.py",
    "backend/app/services/content_service.py",
    "backend/app/services/collaborative_service.py",
    "backend/app/services/hybrid_service.py",

    "backend/app/schemas/__init__.py",
    "backend/app/config.py",

    # ==========================
    # FRONTEND
    # ==========================
    "frontend/.gitkeep",

    # ==========================
    # TESTS
    # ==========================
    "tests/__init__.py",
    "tests/test_content.py",
    "tests/test_collaborative.py",
    "tests/test_hybrid.py",
    "tests/test_api.py",

    # ==========================
    # DOCKER
    # ==========================
    "docker/Dockerfile.backend",
    "docker/Dockerfile.frontend",

    # ==========================
    # GITHUB ACTIONS
    # ==========================
    ".github/workflows/ci.yml",
    ".github/workflows/cd.yml",

    # ==========================
    # ROOT FILES
    # ==========================
    ".gitignore",
    ".env",
    "docker-compose.yml",
    "requirements.txt",
    "setup.py",
    "README.md",
]


for filepath in list_of_files:

    filepath = Path(filepath)

    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w"):
            pass
        logging.info(f"Creating file: {filepath}")

    else:
        logging.info(f"{filepath} already exists")