# Chat_DataBaseV2 🚀  

An **AI-powered chatbot** built with **FastAPI**, **LangChain**, and **Google Gemini**, designed to provide natural language access to the `ClassicModels` SQL database.  
This project is deployed on **Hugging Face Spaces**:  
👉 [Live Demo](https://huggingface.co/spaces/gyesibiney/Chat_DataBaseV2)  

---

## 🔹 Features  

- 💬 **Chat with Database** – Ask questions in plain English, get answers directly from the database.  
- ⚡ **FastAPI Backend** – High-performance API powering the chatbot.  
- 🛡️ **Security** – API key authentication for sensitive endpoints.  
- 📊 **Monitoring & Logging** –  
  - Prometheus metrics (`/metrics`)  
  - Rotating logs for debugging & auditing  
- ⏳ **Rate Limiting** – Prevents overload from too many requests.  
- 📅 **Background Tasks** – Automatic database backup scheduler.  
- 🏥 **Health Check** – `/health` endpoint ensures DB availability.  

---

## 🔹 Tech Stack  

- **FastAPI** – Backend framework  
- **LangChain** – SQL Agent for natural language to SQL  
- **Google Gemini** – LLM powering the chatbot  
- **SQLite (`classicmodels.db`)** – Demo relational database  
- **Prometheus** – Metrics and monitoring  
- **APScheduler** – Background tasks (e.g., DB backups)  
- **Uvicorn** – ASGI server  

---

## 🔹 Endpoints  

| Endpoint        | Method | Description |
|-----------------|--------|-------------|
| `/`             | GET    | Web UI (chat interface) |
| `/chat`         | POST   | Send question to chatbot (JSON: `{"question": "..."}`) |
| `/health`       | GET    | Health check for DB connectivity |
| `/metrics`      | GET    | Prometheus metrics |

---

## 🔹 Example Questions  

Users can try asking:  
- *"Show me the top 5 customers by sales."*  
- *"Which product line has the highest revenue?"*  
- *"How many employees work in the Sales department?"*  
- *"List all orders shipped to France in 2004."*  

---

## 🔹 Deployment  

This project can run in multiple environments:  

### ▶️ Local Run  

```bash
pip install -r requirements.txt
uvicorn main:app --reload

🐳 Docker (for production)

FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]


Build and run:
docker build -t chatbot-db .
docker run -p 7860:7860 chatbot-db


☁️ Cloud / Azure
Can be deployed on Azure App Service or AKS (Kubernetes).

Supports scaling, monitoring, and secret management.








---
title: Chat DataBaseV2
emoji: 🔥
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
```

📌 Author
👤 David Gyesi Biney
🔗 Hugging Face Profile: gyesibiney
