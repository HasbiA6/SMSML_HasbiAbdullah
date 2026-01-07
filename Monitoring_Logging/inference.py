from fastapi import FastAPI, Request
from pydantic import BaseModel, conlist
from starlette.responses import Response, JSONResponse
import time
import os

import psutil
import numpy as np
import mlflow.pyfunc

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="ML Serving with Prometheus Metrics")

PROCESS = psutil.Process(os.getpid())

# =========================
# Load model (REAL)
# =========================
MODEL_URI = os.getenv("MODEL_URI", "/app/model")  # arahkan ke artefak MLflow model/
FEATURE_DIM = int(os.getenv("FEATURE_DIM", "5"))

try:
    MODEL = mlflow.pyfunc.load_model(MODEL_URI)
    MODEL_LOADED = True
except Exception as e:
    MODEL = None
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)

# =========================
# Metrics
# =========================
HTTP_REQUESTS_TOTAL = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
HTTP_REQUEST_DURATION = Histogram("http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"])
HTTP_INPROGRESS = Gauge("http_requests_inprogress", "HTTP requests in progress", ["endpoint"])

INFERENCE_REQUESTS_TOTAL = Counter("inference_requests_total", "Total inference requests")
INFERENCE_ERRORS_TOTAL = Counter("inference_errors_total", "Total inference errors")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Inference latency (s)")

PROCESS_CPU_PERCENT = Gauge("process_cpu_percent", "Process CPU percent")
PROCESS_MEM_RSS = Gauge("process_memory_rss_bytes", "Process RSS memory")
PROCESS_MEM_PERCENT = Gauge("process_memory_percent", "Process memory percent")
PROCESS_THREADS = Gauge("process_threads", "Process threads")

MODEL_PREDICTIONS_TOTAL = Counter("model_predictions_total", "Total predictions produced")
MODEL_PRED_VALUE = Histogram("model_prediction_value", "Prediction value distribution")

MODEL_LOADED_GAUGE = Gauge("model_loaded", "1 if model loaded else 0")
MODEL_LOAD_FAILS = Counter("model_load_failures_total", "Total model load failures")

if MODEL_LOADED:
    MODEL_LOADED_GAUGE.set(1)
else:
    MODEL_LOADED_GAUGE.set(0)
    MODEL_LOAD_FAILS.inc()

# =========================
# Request 
# =========================
class InferenceRequest(BaseModel):
    features: conlist(float, min_length=FEATURE_DIM, max_length=FEATURE_DIM)

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method

    HTTP_INPROGRESS.labels(endpoint=endpoint).inc()
    start = time.time()
    status = "500"

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    finally:
        HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(time.time() - start)
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status).inc()
        HTTP_INPROGRESS.labels(endpoint=endpoint).dec()

@app.get("/health")
def health():
    if MODEL_LOADED:
        return {"status": "ok", "model_uri": MODEL_URI, "feature_dim": FEATURE_DIM}
    return {"status": "degraded", "model_uri": MODEL_URI, "error": MODEL_LOAD_ERROR}

@app.post("/predict")
def predict(req: InferenceRequest):
    INFERENCE_REQUESTS_TOTAL.inc()
    t0 = time.time()

    try:
        if not MODEL_LOADED or MODEL is None:
            INFERENCE_ERRORS_TOTAL.inc()
            return JSONResponse(status_code=500, content={"error": "model not loaded", "model_uri": MODEL_URI})

        x = np.array(req.features, dtype=float).reshape(1, -1)
        y = MODEL.predict(x)

        pred = float(y[0]) if hasattr(y, "__len__") else float(y)

        MODEL_PREDICTIONS_TOTAL.inc()
        MODEL_PRED_VALUE.observe(pred)

        return {"prediction": pred}
    except Exception as e:
        INFERENCE_ERRORS_TOTAL.inc()
        return JSONResponse(status_code=500, content={"error": "inference failed", "detail": str(e)})
    finally:
        INFERENCE_LATENCY.observe(time.time() - t0)

        PROCESS_CPU_PERCENT.set(PROCESS.cpu_percent(interval=None))
        PROCESS_MEM_RSS.set(PROCESS.memory_info().rss)
        PROCESS_MEM_PERCENT.set(PROCESS.memory_percent())
        PROCESS_THREADS.set(PROCESS.num_threads())

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)