import os
import time
import shutil
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict
from datetime import datetime

import sqlite3
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from apscheduler.schedulers.background import BackgroundScheduler

# --- LangChain & Gemini imports ---
import google.generativeai as genai
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------- Configuration ----------
BASE_DIR = os.path.dirname(__file__)
DB_NAME = os.environ.get("DB_NAME", "classicmodels.db")
DB_PATH = os.path.join(BASE_DIR, DB_NAME)

API_KEY = os.environ.get("API_KEY")

RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SEC = int(os.environ.get("RATE_LIMIT_WINDOW_SEC", "60"))

BACKUP_DIR = os.environ.get("BACKUP_DIR", "/tmp/db_backups")
os.makedirs(BACKUP_DIR, exist_ok=True)

LOG_DIR = os.path.join("/tmp", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("classicmodels")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(os.path.join(LOG_DIR, "app.log"), maxBytes=2_000_000, backupCount=3)
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)

REQUEST_COUNTER = Counter("classicmodels_requests_total", "Total number of requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("classicmodels_request_latency_seconds", "Request latency", ["endpoint"])

client_requests: Dict[str, list] = {}

def is_rate_limited(client_id: str) -> bool:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    punches = client_requests.get(client_id, [])
    punches = [t for t in punches if t >= window_start]
    punches.append(now)
    client_requests[client_id] = punches
    return len(punches) > RATE_LIMIT_REQUESTS

if not os.path.exists(DB_PATH):
    for f in os.listdir(BASE_DIR):
        if f.startswith("classicmodels") and f.endswith(".db"):
            src = os.path.join(BASE_DIR, f)
            shutil.copy(src, DB_PATH)
            logger.info(f"Copied DB from {src} to {DB_PATH}")
            break

try:
    conn = sqlite3.connect(DB_PATH)
    tables = conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    conn.close()
    logger.info(f"Database connected. Tables: {[t[0] for t in tables]}")
except Exception as e:
    logger.exception("Database error")
    raise RuntimeError(f"Database error: {str(e)}")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("No GEMINI_API_KEY set in environment")

llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY,
    max_output_tokens=2048,
    top_k=40,
    top_p=0.95
)

db = SQLDatabase.from_uri(
    f"sqlite:///{DB_PATH}",
    include_tables=[
        'productlines', 'products', 'offices',
        'employees', 'customers', 'payments',
        'orders', 'orderdetails'
    ],
    sample_rows_in_table_info=1,
    view_support=False
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a ClassicModels database expert. Follow these rules:
1. Use these relationships:
   - customers → orders → orderdetails → products → productlines
   - employees → offices
   - customers → payments
2. Format currency as USD ($1,000.00)
3. Use dates as YYYY-MM-DD
4. Never modify data
5. Schema: {schema}"""),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")
])

agent = create_sql_agent(
    llm=llm_model,
    db=db,
    prompt=prompt,
    agent_type="openai-tools",
    verbose=False,
    max_iterations=10,
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

app = FastAPI(
    title="ClassicModels Database Assistant",
    description="FastAPI + Gemini + LangChain SQL-agent example with MCO features",
    version="1.0"
)

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
if os.path.isdir(os.path.join(BASE_DIR, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.middleware("http")
async def protect_docs_and_rate_limit(request: Request, call_next):
    path = request.url.path
    client = request.client.host if request.client else "unknown"

    if is_rate_limited(client):
        REQUEST_COUNTER.labels(method=request.method, endpoint=path, http_status="429").inc()
        return PlainTextResponse("Rate limit exceeded", status_code=status.HTTP_429_TOO_MANY_REQUESTS)

    if API_KEY:
        protected_paths = ["/docs", "/redoc", "/openapi.json", "/query", "/"]
        if any(path.startswith(p) for p in protected_paths):
            header_key = request.headers.get("x-api-key")
            if header_key != API_KEY:
                REQUEST_COUNTER.labels(method=request.method, endpoint=path, http_status="401").inc()
                return PlainTextResponse("Unauthorized - provide X-API-KEY header", status_code=status.HTTP_401_UNAUTHORIZED)

    start = time.time()
    response = await call_next(request)
    latency = time.time() - start
    REQUEST_LATENCY.labels(endpoint=path).observe(latency)
    REQUEST_COUNTER.labels(method=request.method, endpoint=path, http_status=str(response.status_code)).inc()
    return response

def process_query(question: str) -> str:
    try:
        blocked_terms = ["drop", "delete", "insert", "update", "alter", ";--"]
        if any(term in question.lower() for term in blocked_terms):
            raise ValueError("Data modification queries are disabled")

        schema = db.get_table_info()
        response = agent.invoke({
            "input": question,
            "schema": schema
        })
        result = response.get('output', "")
        if "```sql" in result:
            parts = result.split("```")
            for p in reversed(parts):
                if "sql" in p:
                    result = p.replace("sql", "").strip()
                    break
        return result
    except Exception as e:
        logger.exception("Query processing error")
        raise

class QueryRequest(BaseModel):
    question: str

# **Example questions added here**
EXAMPLE_QUESTIONS = [
    "Show me all products in the 'Classic Cars' product line.",
    "List customers who placed orders in 2024-01.",
    "What are the total payments made by customer 'John Doe'?",
    "Give me details of employees in the 'Sales' office.",
    "How many orders were made last month?"
]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "api_key_required": bool(API_KEY),
        "example_questions": EXAMPLE_QUESTIONS
    })

@app.post("/query")
async def query_db(req: QueryRequest):
    if len(req.question) > 2000:
        raise HTTPException(status_code=413, detail="Question too long")
    try:
        ans = process_query(req.question)
        return {"result": ans}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1;")
        conn.close()
        return {"status": "ok", "db": os.path.exists(DB_PATH)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

def backup_db_job():
    try:
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup_path = os.path.join(BACKUP_DIR, f"classicmodels_{ts}.db")
        shutil.copy(DB_PATH, backup_path)
        logger.info(f"DB backup created: {backup_path}")
    except Exception:
        logger.exception("DB backup failed")

scheduler = BackgroundScheduler()
scheduler.add_job(backup_db_job, "cron", hour=3)
scheduler.start()

@app.on_event("shutdown")
def shutdown_event():
    try:
        scheduler.shutdown(wait=False)
    except Exception:
        pass
