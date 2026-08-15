# Project Progress & Architecture

This repository tracks the ML Engineering Platform for the Zindi Financial Stress Prediction challenge.

## Machine Roles

We operate across two distinct environments to prevent resource constraints on development machines.

**Developer Machine (Local):**
- Coding and refactoring
- Lightweight tests (e.g. `pytest`)
- API / Dashboard development & running via Docker
- Documentation and architecture planning
- Git operations

**Training Machine (Remote/Powerful):**
- Full model training (`python train.py`)
- Cross-validation
- Inference (`python predict.py`)
- Experiment JSON generation
- Expensive evaluation and metric calculation

## Task Status

| Component | Status |
| :--- | :--- |
| Data ingestion | COMPLETED |
| Data cleaning | COMPLETED |
| Feature engineering | COMPLETED |
| Cross-validation | COMPLETED |
| Model training | COMPLETED |
| OOF prediction | COMPLETED |
| Ensembling | COMPLETED |
| Submission generation | COMPLETED |
| Model persistence | COMPLETED |
| Inference pipeline | COMPLETED |
| Experiment tracking | COMPLETED |
| Evaluation metrics | COMPLETED |
| Dashboard | COMPLETED |
| API | COMPLETED |
| Testing | COMPLETED |
| Docker | COMPLETED |
| Documentation | COMPLETED |
| Project Management Directives | COMPLETED |
| Explainability | PLANNED |
| Deployment | PLANNED |
