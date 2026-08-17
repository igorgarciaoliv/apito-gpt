# -*- coding: utf-8 -*-
"""
APITO GPT — API REST (FastAPI)
Executa: uvicorn api:app --host 0.0.0.0 --port 8000
"""
import uvicorn
from app import app  # FastAPI app definido em app.py

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
