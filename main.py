# app_fastapi.py
import os
import shutil
import sqlite3
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangChain / Gemini imports (same as your original)
import google.generativeai as genai  # keep if you need it; not actively used below
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

DB_NAME = "classicmodels.db"

# Ensure DB file present (same logic as your script)
if not os.path.exists(DB_NAME):
    for file in os.listdir():
        if file.startswith("classicmodels") and file.endswith(".db"):
            shutil.copy(file, DB_NAME)
            print(f"Using database file: {file}")
            break

# verify DB connection at startup
try:
    conn = sqlite3.connect(DB_NAME)
    tables = conn.cursor().execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    conn.close()
    print(f"Database connected. Tables: {[t[0] for t in tables]}")
except Exception as e:
    print(f"Database error: {str(e)}")
    raise

# Setup SQLDatabase wrapper (used by agent)
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

# Load Gemini API key from env
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("No GEMINI_API_KEY found. Set the environment variable before starting the app.")

# Initialize the LLM wrapper (same config as your original)
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.3,
    google_api_key=GEMINI_API_KEY,
    max_output_tokens=2048,
    top_k=40,
    top_p=0.95
)

# Prompt template (same as original)
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

# Create the agent. This can be expensive — create once.
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

# FastAPI app
app = FastAPI(title="ClassicModels Database Assistant (FastAPI)")

# Allow cross-origin requests (useful for local index.html or other frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in prod!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class QueryRequest(BaseModel):
    question: str

# Safety blocking set
BLOCKED_TERMS = {"drop", "delete", "insert", "update", "alter", ";--"}

def is_blocked(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in BLOCKED_TERMS)

@app.post("/api/query")
async def api_query(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    if is_blocked(question):
        raise HTTPException(status_code=403, detail="Data modification queries are disabled for safety.")

    try:
        # Provide schema info to the agent (as in your script)
        schema = db.get_table_info()

        response = agent.invoke({
            "input": question,
            "schema": schema
        })

        result = response.get("output", "")

        # Clean code fences/artifacts if any
        if "```sql" in result:
            # extract last SQL block (if provided)
            parts = result.split("```")
            # find SQL block if any
            for p in reversed(parts):
                if "sql" in p:
                    result = p.replace("sql", "").strip()
                    break

        return {"success": True, "answer": result}

    except Exception as e:
        # Keep error user-friendly
        return {
            "success": False,
            "error": str(e),
            "hint": "Try simpler queries like: 'Show customers from France' or 'List products needing restock'."
        }

@app.get("/health")
async def health():
    return {"status": "ok"}

# Optionally provide a minimal root page for quick manual testing (JSON reply)
@app.get("/")
async def root():
    return {
        "service": "ClassicModels Database Assistant (FastAPI)",
        "endpoints": {
            "POST /api/query": {"body": {"question": "string"}},
            "GET /health": {}
        }
    }


