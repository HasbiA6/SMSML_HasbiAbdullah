from fastapi import FastAPI, Request
from pydantic import BaseModel
from starlette.responses import Response, JSONResponse
import time
import random
import os

import psutil
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)

app = FastAPI(title="ML Serving with Prometheus Metrics")

PROCESS = psutil.Process(os.getpid())

# -------------------------
# HTTP-level metrics
# -------------------------
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"]
)

HTTP_INPROGRESS = Gauge(
    "http_requests_inprogress",
    "Number of HTTP requests in progress",
    ["endpoint"]
)

# -------------------------
# Inference metrics
# -------------------------
INFERENCE_REQUESTS_TOTAL = Counter(
    "inference_requests_total",
    "Total number of inference requests"
)

INFERENCE_ERRORS_TOTAL = Counter(
    "inference_errors_total",
    "Total number of inference errors"
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency in seconds"
)

MODEL_PREDICTIONS_TOTAL = Counter(
    "model_predictions_total",
    "Total predictions produced"
)

MODEL_PRED_VALUE = Histogram(
    "model_prediction_value",
    "Distribution of prediction values"
)

# -------------------------
# Process metrics (extra)
# -------------------------
PROCESS_CPU_PERCENT = Gauge("process_cpu_percent", "Process CPU percent")
PROCESS_MEM_RSS = Gauge("process_memory_rss_bytes", "Process RSS memory in bytes")
PROCESS_MEM_PERCENT = Gauge("process_memory_percent", "Process memory percent")
PROCESS_THREADS = Gauge("process_threads", "Number of process threads")


class InferenceRequest(BaseModel):
    features: list[float]


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    endpoint = request.url.path
    method = request.method

    HTTP_INPROGRESS.labels(endpoint=endpoint).inc()
    start = time.time()

    try:
        response = await call_next(request)
        status = str(response.status_code)
        return response
    except Exception:
        status = "500"
        raise
    finally:
        duration = time.time() - start
        HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
        HTTP_REQUESTS_TOTAL.labels(method=method, endpoint=endpoint, status=status).inc()
        HTTP_INPROGRESS.labels(endpoint=endpoint).dec()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: InferenceRequest):
    INFERENCE_REQUESTS_TOTAL.inc()
    t0 = time.time()

    try:
        # simulasi inference (ganti dengan modelmu)
        time.sleep(random.uniform(0.03, 0.15))
        pred = float(sum(req.features))

        MODEL_PREDICTIONS_TOTAL.inc()
        MODEL_PRED_VALUE.observe(pred)

        return {"prediction": pred}
    except Exception:
        INFERENCE_ERRORS_TOTAL.inc()
        return JSONResponse(status_code=500, content={"error": "inference failed"})
    finally:
        INFERENCE_LATENCY.observe(time.time() - t0)

        # update process metrics setiap request (simple & aman)
        PROCESS_CPU_PERCENT.set(PROCESS.cpu_percent(interval=None))
        PROCESS_MEM_RSS.set(PROCESS.memory_info().rss)
        PROCESS_MEM_PERCENT.set(PROCESS.memory_percent())
        PROCESS_THREADS.set(PROCESS.num_threads())


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)