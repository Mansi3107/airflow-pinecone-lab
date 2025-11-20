# Airflow + Pinecone Semantic Search Pipeline

This project demonstrates how to run a Pinecone semantic search job inside an Apache Airflow DAG.  
It was created as part of **Homework 8: Running Pinecone Job as an Airflow Pipeline**.

---

## 📘 Overview
The pipeline automates an end-to-end workflow:
1. Generate an input text file (`prepare_input`)
2. Create or verify a Pinecone index (`create_index`)
3. Convert text into embeddings and upload them (`embed_and_upsert`)
4. Run a semantic search query against the Pinecone index (`query_index`)

All steps run automatically inside **Dockerized Airflow** containers.

---

## ⚙️ Environment Setup

### Docker Compose
Airflow services (webserver, scheduler, worker, Redis, Postgres) are managed using Docker Compose.  
The `docker-compose.yaml` file was modified to include additional Python packages:

```yaml
_PIP_ADDITIONAL_REQUIREMENTS: >-
  sentence-transformers
  pinecone>=4.0.0
