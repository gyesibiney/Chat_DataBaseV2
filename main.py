# main.py
import os
import time
import shutil
import sqlite3
from typing import Optional, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
import psutil

# Prometheus client
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST, Gauge

# LangChain & Gemini imports (as used in your project)
import google.generativeai as genai
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -------------------------
# Configuration / Setup
# -------------------------
APP_START = time.time()
BASE_DIR = os.path.dirname(__file__) or "."
DB_NAME = os.environ.get("DB_NAME", "classicmodels.db")
DB_PATH = os.path.join(BASE_DIR, DB_NAME)

# Try to ensure DB present (copy similarly named file from repo root if needed)
if not os.path.exists(DB_PATH):
    for f in os.listdir(BASE_DIR):
        if f.startswith("classicmodels") and f.endswith(".db"):
            shutil.copy(os.path.join(BASE_DIR, f), DB_PATH)
            print(f"Copied DB from {f} to {DB_PATH}")
            break

# quick DB check (fail fast if missing)
try:
    conn = sqlite3.connect(DB_PATH)
    tables = conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    conn.close()
    print(f"Database connected. Tables: {[t[0] for t in tables]}")
except Exception as e:
    raise RuntimeError(f"Database error: {str(e)}")

# -------------------------
# Initialize LLM + Agent
# -------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY must be set in the environment (Hugging Face Space secret).")

llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY,
    max_output_tokens=2048,
    top_k=40,
    top_p=0.95
)

# SQLDatabase wrapper used by the agent
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

# -------------------------
# Instrumentation (Prometheus)
# -------------------------
REQUEST_COUNTER = Counter("classicmodels_requests_total", "Total HTTP requests", ["method", "endpoint", "http_status"])
REQUEST_LATENCY = Histogram("classicmodels_request_latency_seconds", "Request latency seconds", ["endpoint"])
IN_FLIGHT = Gauge("classicmodels_in_flight_requests", "Currently in-flight requests")

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(title="ClassicModels Database Assistant", version="1.0")

# -------------------------
# Core query logic (safe)
# -------------------------
def process_query(question: str) -> str:
    """
    Call the LangChain agent to translate/execute the question against the DB.
    Blocks dangerous modification queries.
    Returns agent output text (cleaned of fenced SQL if present).
    """
    blocked_terms = ["drop", "delete", "insert", "update", "alter", ";--"]
    if any(term in question.lower() for term in blocked_terms):
        raise ValueError("Data modification queries are disabled")

    schema = db.get_table_info()
    response = agent.invoke({
        "input": question,
        "schema": schema
    })
    result = response.get("output", "") if isinstance(response, dict) else str(response)

    if "```" in result:
        # try to extract the last fenced block or SQL block
        parts = result.split("```")
        for p in reversed(parts):
            if p.strip():
                # Remove a leading "sql" marker
                if p.strip().lower().startswith("sql"):
                    cleaned = p.replace("sql", "", 1).strip()
                    result = cleaned
                else:
                    result = p.strip()
                break
    return result

# -------------------------
# Helper: time+instrument decorator-like usage
# -------------------------
def instrument(endpoint: str, method: str = "GET"):
    class _Ctx:
        def __enter__(self):
            IN_FLIGHT.inc()
            self._start = time.time()
        def __exit__(self, exc_type, exc, tb):
            dur = time.time() - self._start
            REQUEST_LATENCY.labels(endpoint=endpoint).observe(dur)
            IN_FLIGHT.dec()
    return _Ctx()

# -------------------------
# Routes
# -------------------------
class QueryRequest(BaseModel):
    question: str

@app.get("/", response_class=HTMLResponse)
async def ui_root():
    """
    Inline HTML UI. Uses fetch to POST /query with JSON.
    """
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ClassicModels Assistant</title>
  <style>
    body{{font-family:Inter,system-ui,Arial;background:#f7fafc;color:#0f172a;margin:0;padding:24px}}
    .container{{max-width:980px;margin:0 auto}}
    .card{{background:#fff;padding:20px;border-radius:10px;box-shadow:0 6px 18px rgba(2,6,23,0.06)}}
    h1{{margin:0 0 6px;font-size:22px}}
    .muted{{color:#6b7280;font-size:13px}}
    input[type=text]{{width:100%;padding:10px;border-radius:8px;border:1px solid #e6e9ef;margin-top:8px}}
    button{{background:#2563eb;color:#fff;border:0;padding:10px 14px;border-radius:8px;cursor:pointer}}
    pre.result{{white-space:pre-wrap;background:#f8fafc;border:1px solid #eef2ff;padding:12px;border-radius:8px;margin-top:12px}}
    .links a{{margin-right:8px;color:#374151;text-decoration:none;font-size:14px}}
    .small{{font-size:13px;color:#64748b}}
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div>
          <h1>🏭 ClassicModels Database Assistant</h1>
          <div class="muted">Ask natural language questions — powered by Gemini + LangChain</div>
        </div>
        <div class="links">
          <a href="/docs">Swagger</a>
          <a href="/redoc">ReDoc</a>
          <a href="/health">Health</a>
          <a href="/metrics">Metrics</a>
          <a href="/system-metrics">System</a>
        </div>
      </div>

      <div style="margin-top:16px">
        <label class="small">Question</label>
        <input id="q" type="text" placeholder="e.g., Show customers from France with >5 orders" />
        <div style="display:flex;gap:8px;margin-top:8px">
          <input id="apikey" type="password" placeholder="X-API-KEY (if required)" style="flex:1;padding:10px;border-radius:8px;border:1px solid #e6e9ef"/>
          <button onclick="ask()">Ask</button>
        </div>
        <div id="out" class="result" style="display:none"></div>
      </div>

      <div style="margin-top:14px" class="small">Tip: keep questions short and focused on customers, products, orders, or employees.</div>
    </div>
  </div>

<script>
async function ask(){
  const q = document.getElementById("q").value.trim();
  const key = document.getElementById("apikey").value.trim();
  const out = document.getElementById("out");
  out.style.display = "block";
  out.textContent = "Thinking...";

  try{
    const headers = {"Content-Type":"application/json"};
    if(key) headers["X-API-KEY"] = key;
    const res = await fetch("/query", {method:"POST", headers, body: JSON.stringify({question: q})});
    const data = await res.json();
    if(res.ok){
      out.textContent = data.result;
    } else {
      out.textContent = "Error " + res.status + ": " + (data.detail || JSON.stringify(data));
    }
  } catch(e){
    out.textContent = "Request failed: " + String(e);
  }
}
</script>
</body>
</html>"""
    # instrument
    REQUEST_COUNTER.labels(method="GET", endpoint="/", http_status="200").inc()
    return HTMLResponse(content=html)

@app.post("/query")
async def query(req: QueryRequest, request: Request):
    start = time.time()
    IN_FLIGHT.inc()
    try:
        if not req.question or not req.question.strip():
            raise HTTPException(status_code=400, detail="Empty question")
        # Process query (may call LLM)
        answer = process_query(req.question)
        REQUEST_COUNTER.labels(method="POST", endpoint="/query", http_status="200").inc()
        return {"result": answer}
    except HTTPException as he:
        REQUEST_COUNTER.labels(method="POST", endpoint="/query", http_status=str(he.status_code)).inc()
        raise he
    except Exception as e:
        REQUEST_COUNTER.labels(method="POST", endpoint="/query", http_status="500").inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        REQUEST_LATENCY.labels(endpoint="/query").observe(time.time() - start)
        IN_FLIGHT.dec()

@app.get("/health")
async def health():
    # quick DB check
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1;")
        conn.close()
        REQUEST_COUNTER.labels(method="GET", endpoint="/health", http_status="200").inc()
        return {"status":"ok"}
    except Exception as e:
        REQUEST_COUNTER.labels(method="GET", endpoint="/health", http_status="500").inc()
        return JSONResponse(status_code=500, content={"status":"error","detail":str(e)})

@app.get("/system-metrics")
async def system_metrics():
    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    uptime = int(time.time() - APP_START)
    REQUEST_COUNTER.labels(method="GET", endpoint="/system-metrics", http_status="200").inc()
    return {
        "cpu_percent": cpu,
        "memory_percent": mem.percent,
        "memory_total": mem.total,
        "disk_percent": disk.percent,
        "uptime_seconds": uptime
    }

@app.get("/metrics")
async def metrics():
    """
    Prometheus exposition format (scrapable by Prometheus).
    Includes counters/histograms we've defined.
    """
    REQUEST_COUNTER.labels(method="GET", endpoint="/metrics", http_status="200").inc()
    data = generate_latest()
    return PlainTextResponse(data, media_type=CONTENT_TYPE_LATEST)

# If run directly (useful for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), reload=True)
