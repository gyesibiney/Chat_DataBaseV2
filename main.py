import os
import shutil
import sqlite3
import psutil
import time
from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
import google.generativeai as genai
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import uvicorn

# ---------- 1. Database setup ----------
DB_NAME = "classicmodels.db"

if not os.path.exists(DB_NAME):
    for file in os.listdir():
        if file.startswith("classicmodels") and file.endswith(".db"):
            shutil.copy(file, DB_NAME)
            print(f"Using database file: {file}")
            break

try:
    conn = sqlite3.connect(DB_NAME)
    tables = conn.cursor().execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    ).fetchall()
    conn.close()
    print(f"Database connected. Tables: {[t[0] for t in tables]}")
except Exception as e:
    raise RuntimeError(f"Database error: {str(e)}")

# ---------- 2. LangChain + Gemini setup ----------
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
    raise ValueError("No Gemini API key found. Please set GEMINI_API_KEY.")

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

# ---------- 3. Query processing ----------
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
        result = response['output']

        if "```sql" in result:
            result = result.split("```")[-2].replace("```sql", "").strip()
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------- 4. FastAPI app ----------
app = FastAPI(title="ClassicModels Database Assistant", version="1.1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static & templates dirs exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Static files & templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    question: str

# API Endpoint
@app.post("/query")
async def query_db(req: QueryRequest):
    answer = process_query(req.question)
    return {"result": answer}

# Web Interface
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request, question: str = Form(...)):
    result = process_query(question)
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "question": question})

# ---------- 5. Monitoring Endpoints ----------
@app.get("/health", tags=["Monitoring"])
def health_check():
    return {"status": "ok", "message": "API is running"}

@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    process = psutil.Process()
    uptime = time.time() - process.create_time()
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "uptime_seconds": round(uptime, 2)
    }

# ---------- 6. Custom OpenAPI for nicer docs ----------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ClassicModels Database Assistant",
        version="1.1.0",
        description="FastAPI app with Gemini LLM, SQLite, UI, and monitoring endpoints.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# ---------- 7. Hugging Face entry ----------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)

