from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from .dependencies import (
    get_project_status,
    get_all_experiments,
    get_latest_experiment,
    get_all_models,
    get_submissions
)

app = FastAPI(title="Zindi ML Engineering API", version="1.0.0")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/project/status")
def project_status():
    status = get_project_status()
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return JSONResponse(content=status)

@app.get("/api/models")
def list_models():
    models = get_all_models()
    return {"models": list(models.keys())}

@app.get("/api/models/{model_name}")
def model_details(model_name: str):
    models = get_all_models()
    if model_name not in models:
        raise HTTPException(status_code=404, detail="Model not found in experiment history.")
    return {"model": model_name, "history": models[model_name]}

@app.get("/api/experiments")
def list_experiments():
    return {"experiments": get_all_experiments()}

@app.get("/api/experiments/latest")
def latest_experiment():
    exp = get_latest_experiment()
    if not exp:
        raise HTTPException(status_code=404, detail="No experiments found.")
    return exp

@app.get("/api/submissions")
def list_submissions():
    return {"submissions": get_submissions()}
