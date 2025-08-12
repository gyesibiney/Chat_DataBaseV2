import os
import sqlite3
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Histogram

# -------------------
# Ensure static & templates directories exist
# -------------------
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Create placeholder CSS if missing
if not os.path.exists("static/css/style.css"):
    with open("static/css/style.css", "w") as f:
        f.write("body { font-family: Arial, sans-serif; margin: 20px; background: #f4f4f4; } h1 { color: #333; }")

# -------------------
# FastAPI App
# -------------------
app = FastAPI(title="ClassicModels Assistant", description="Nice FastAPI interface with metrics")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

# -------------------
# Prometheus Metrics
# -------------------
REQUEST_COUNT = Counter("app_requests_total", "Total number of requests", ["method", "endpoint"])
REQUEST_LATENCY = Histogram("app_request_latency_seconds", "Request latency", ["endpoint"])

# -------------------
# Routes
# -------------------
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    with REQUEST_LATENCY.labels(endpoint="/").time():
        return templates.TemplateResponse("index.html", {"request": request, "title": "ClassicModels DB"})

@app.get("/customers")
def get_customers():
    REQUEST_COUNT.labels(method="GET", endpoint="/customers").inc()
    with REQUEST_LATENCY.labels(endpoint="/customers").time():
        conn = sqlite3.connect("classicmodels.db")
        cursor = conn.cursor()
        cursor.execute("SELECT customerNumber, customerName, country FROM customers LIMIT 10;")
        rows = cursor.fetchall()
        conn.close()
        return {"customers": rows}

@app.get("/metrics")
def metrics():
    return HTMLResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# -------------------
# Create example template if missing
# -------------------
index_html_path = "templates/index.html"
if not os.path.exists(index_html_path):
    with open(index_html_path, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <h1>{{ title }}</h1>
    <p>Welcome to the ClassicModels database interface.</p>
    <button onclick="loadCustomers()">Load Customers</button>
    <div id="output"></div>
    <script>
        async function loadCustomers() {
            let res = await fetch('/customers');
            let data = await res.json();
            let html = "<ul>";
            data.customers.forEach(c => {
                html += `<li>${c[1]} (${c[2]})</li>`;
            });
            html += "</ul>";
            document.getElementById("output").innerHTML = html;
        }
    </script>
</body>
</html>
""")
