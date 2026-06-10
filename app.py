#!/usr/bin/env python3
"""
app.py — WxSmith gridded-snowfall module as a FastAPI service.

This is the "thin product on a shared platform" pattern: a FastAPI router + a
frontend module over the analysis engine. Endpoints:

  GET  /            -> the worksheet UI (index.html)
  GET  /api/schema  -> the conditional worksheet schema (drives the form)
  POST /api/run     -> {answers} -> validate + execute + render -> result + figure

Run:
  pip install fastapi uvicorn
  WXAGENT_DATA_DIR=/path/to/sfav2 uvicorn app:app --reload --port 8000
  open http://localhost:8000
"""
import os
import matplotlib
matplotlib.use("Agg")
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

from analyses import build_analysis_worksheet
from service import run_analysis

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("WXAGENT_DATA_DIR", ".")

app = FastAPI(title="WxSmith — Gridded Snowfall Analysis")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "index.html")) as f:
        return f.read()


@app.get("/api/schema")
def schema():
    return build_analysis_worksheet()


@app.get("/api/years")
def years():
    from analyses import find_sfav2_files
    ys = sorted(find_sfav2_files(DATA_DIR).keys())
    return {"years": ys}


@app.post("/api/run")
def run(answers: dict):
    try:
        return run_analysis(answers, DATA_DIR)
    except ValidationError as e:
        clean = [{"loc": list(err.get("loc", ())), "msg": err.get("msg", ""),
                  "type": err.get("type", "")} for err in e.errors()]
        raise HTTPException(status_code=422, detail=clean)
    except (KeyError, ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))
