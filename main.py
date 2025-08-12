# main.py
import os
import time
import sqlite3
import shutil
import psutil
from typing import Optional

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Prometheus client
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# LangChain / Gemini imports (your existing logic)
import google.generativeai as genai
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# -------------------------
# Ensure static & templates exist (auto-create)
# -------------------------
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# small default CSS
if not os.path.exists("static/css/style.css"):
    with open("static/css/style.css", "w") as f:
        f.write("""
body { font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background: #f7fafc; color: #1f2937; padding: 24px; }
.container { max-width: 900px; margin: 0 auto; }
.card { background: white; padding: 18px; border-radius: 8px; box-shadow: 0 4px 10px rgba(2,6,23,0.06); }
.header { display:flex; gap:12px; align-items:center; justify-content:space-between; margin-bottom:16px; }
h1 { margin: 0; font-size: 24px; color:#0f172a; }
input[type="text"], textarea { width:100%; padding:10px; border:1px solid #e6e9ef; border-radius:6px; }
button { background:#2563eb; color:white; border:none; padding:10px 14px; border-radius:6px; cursor:pointer; }
.small { font-size:13px; color:#6b7280; }
.result { white-space: pre-wrap; background:#f8fafc; border:1px solid #eef2ff; padding:12px; border-radius:6px; margin-top:12px; }
.links a { margin-right:10px; color:#374151; text-decoration:none; }
""")

# default template
index_path = "templates/index.html"
if not os.path.exists(index_path):
    with open(index_path, "w") as f:
        f.write("""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>ClassicModels Assistant</title>
  <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
  <div class="container">
    <div class="card">
      <div class="header">
        <div>
          <h1>🏭 ClassicModels Database Assistant</h1>
          <div class="small">Ask natural language questions — powered by Gemini + LangChain</div>
        </div>
        <div class="links">
          <a href="/docs">Swagger</a>
          <a href="/redoc">ReDoc</a>
          <a href="/health">Health</a>
          <a href="/metrics">Metrics</a>
          <a href="/system-metrics">System Metrics</a>
        </div>
      </div>

      <form id="qform" method="post" action="/ask">
        <label class="small">Your question</label>
        <input id="question" name="question" type="text" placeholder="e.g., Show customers from Paris with >5 orders" required />
        <div style="margin-top:8px; display:flex; gap:8px;">
          <input id="apikey" name="apikey" type="password" placeholder="X-API-KEY (if required)" style="flex:1"/>
          <button type="submit">Ask</button>
        </div>
      </form>

      {% if result %}
        <h3 style="margin-top:16px;">Result</h3>
        <div class="result">{{ result }}</div>
      {% endif %}

      <div style="margin-top:16px;" class="small">Tip: use clear, short questions focused on customers, products, orders, or employees.</div>
    </div>
  </div>
</body>
</html>""")

# -------------------------
# App initialization
# -------------------------
app = FastAPI(title="ClassicModels Database Assistant", version="1.2.0")

# Mount static + templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------------
# Prometheus metrics
# -------------------------
REQUEST_COUNTER = Counter("classicmodels_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_LATENCY = Histogram("classicmodels_request_latency_seconds", "Request latency seconds", ["endpoint"])

def instrument_endpoint(endpoint: str, method: str = "GET"):
    """helper decorator-like usage inside handlers (manual instrumentation)."""
    REQUEST_COUNTER.labels(method=method, endpoint=endpoint, status="ok").inc()

# -------------------------
# Database (same as your original)
# -------------------------
DB_NAME = "classicmodels.db"
if not os.path.exists(DB_NAME):
    # attempt to copy a similarly-named db if present
    for file in os.listdir():
        if file.startswith("classicmodels") and file.endswith(".db"):
            shutil.copy(file, DB_NAME)
            print(f"Using database file: {file}")
            break

try:
    conn = sqlite3.connect(DB_NAME)
    tables = conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    conn.close()
    print(f"Database connected. Tables: {[t[0] for t in tables]}")
except Exception as e:
    raise RuntimeError(f"Database error: {str(e)}")

# -------------------------
# LangChain + Gemini setup (unchanged)
# -------------------------
db = SQLDatabase.from_uri(
    f"sqlite:///{DB_NAME}",
    include_tables=[
        'productlines', 'products', 'offices',
        'employees', 'customers', 'payments',
        'orders', 'orderdetails'
    ],
    sample_rows_in_table_info=1,
    view_support=False
)

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
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
# Core query logic (unchanged)
# -------------------------
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
            # extract last SQL block or the main content
            parts = result.split("```")
            # find last non-empty block that isn't plain text wrapper
            for p in reversed(parts):
                if p.strip():
                    if p.strip().startswith("sql"):
                        result = p.replace("sql", "").strip()
                    else:
                        # keep as-is if not marked
                        result = p.strip()
                    break
        return result
    except Exception as e:
        # propagate as HTTPException when called from endpoints
        raise HTTPException(status_code=400, detail=str(e))

# -------------------------
# Request model for /query
# -------------------------
class QueryRequest(BaseModel):
    question: str

# -------------------------
# Endpoints
# -------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    start = time.time()
    resp = templates.TemplateResponse("index.html", {"request": request, "result": None})
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start)
    REQUEST_COUNTER.labels(method="GET", endpoint="/", status=str(resp.status_code)).inc()
    return resp

@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...), apikey: Optional[str] = Form(None)):
    start = time.time()
    try:
        result = process_query(question)
    except HTTPException as e:
        result = f"Error: {e.detail}"
    resp = templates.TemplateResponse("index.html", {"request": request, "result": result})
    REQUEST_LATENCY.labels(endpoint="/ask").observe(time.time() - start)
    REQUEST_COUNTER.labels(method="POST", endpoint="/ask", status=str(resp.status_code)).inc()
    return resp

@app.post("/query")
async def query_db(payload: QueryRequest):
    start = time.time()
    answer = process_query(payload.question)
    REQUEST_LATENCY.labels(endpoint="/query").observe(time.time() - start)
    REQUEST_COUNTER.labels(method="POST", endpoint="/query", status="200").inc()
    return {"result": answer}

@app.get("/health")
async def health():
    # quick DB check
    try:
        conn = sqlite3.connect(DB_NAME)
        conn.execute("SELECT 1;")
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/metrics")
async def prometheus_metrics():
    # returns Prometheus exposition format
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/system-metrics")
async def system_metrics():
    cpu = psutil.cpu_percent(interval=0.2)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage("/")
    return {
        "cpu_percent": cpu,
        "memory_percent": mem.percent,
        "memory_total": mem.total,
        "disk_percent": disk.percent,
        "uptime_seconds": round(time.time() - psutil.boot_time(), 2)
    }

# -------------------------
# Optional: customize OpenAPI info (keeps /docs and /redoc)
# -------------------------
from fastapi.openapi.utils import get_openapi
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ClassicModels Database Assistant",
        version="1.2.0",
        description="FastAPI + Gemini + LangChain SQL-agent with a friendly UI and monitoring endpoints.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# -------------------------
# If you run locally with `python main.py`, start uvicorn
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
