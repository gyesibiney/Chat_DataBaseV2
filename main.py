from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import google.generativeai as genai
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os
import sqlite3
import shutil

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
app = FastAPI(title="ClassicModels Database Assistant")

# Static + Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_db(req: QueryRequest):
    answer = process_query(req.question)
    return {"result": answer}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
