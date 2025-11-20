# pinecone_pipeline_demo.py
# End-to-end Airflow demo: create (or reuse) Pinecone index, embed + upsert, then query.

from datetime import datetime, timedelta
from time import sleep
import os

from airflow import DAG
from airflow.models import Variable
from airflow.operators.python import PythonOperator

# Pinecone 4.x client
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# all-MiniLM-L6-v2 -> 384-dim vectors
INDEX_DIM = 384


# ---------- helpers ----------
def _vars():
    """Read and sanitize Airflow Variables (strip hidden spaces/newlines)."""
    api_key = Variable.get("PINECONE_API_KEY").strip()
    index_name = Variable.get("PINECONE_INDEX_NAME", default_var="airflow-pinecone-demo").strip()
    cloud = Variable.get("PINECONE_CLOUD", default_var="aws").strip()
    region = Variable.get("PINECONE_REGION", default_var="us-east-1").strip()
    return api_key, index_name, cloud, region


# ---------- task fns ----------
def prepare_input_fn(**ctx):
    """
    Create a tiny input text file inside the worker container and push its path via XCom.
    """
    base = "/opt/airflow/include"
    path = os.path.join(base, "sample.txt")
    os.makedirs(base, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "Airflow orchestrates workflows.\n"
            "Pinecone is a vector database for semantic search.\n"
            "Sentence-Transformers turn text into embeddings.\n"
            "This lab runs a Pinecone job from an Airflow DAG.\n"
        )
    print(f"[prepare_input] Created file at: {path}")
    ctx["ti"].xcom_push(key="input_path", value=path)
    print("[prepare_input] XCom pushed successfully.")


def create_index_fn(**_):
    """
    Ensure the Pinecone index exists in the configured cloud/region, and wait until it's ready.
    Logs actual host, project and dimension so downstream tasks are deterministic.
    """
    api_key, index_name, cloud, region = _vars()
    pc = Pinecone(api_key=api_key)

    # 1) Try to use an existing index (e.g., created in console)
    try:
        desc = pc.describe_index(index_name)
        print(f"[create_index] index already exists: {index_name}")
    except Exception as e:
        # Not found or other 404-like error -> create it now
        print(f"[create_index] describe failed ({e.__class__.__name__}): {e}")
        print(f"[create_index] creating index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=INDEX_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        desc = pc.describe_index(index_name)

    # 2) Wait until the index is ready
    print("[create_index] waiting for index to be ready...")
    for _ in range(30):  # ~60s max
        if pc.describe_index(index_name).get("status", {}).get("ready"):
            break
        sleep(2)
    if not pc.describe_index(index_name).get("status", {}).get("ready"):
        raise RuntimeError("Index did not become ready in time")

    # 3) Log the details and enforce the expected dimension
    d = pc.describe_index(index_name)
    dim = d.get("dimension")
    host = d.get("host")
    proj = d.get("projectName")
    print(f"[create_index] READY: name={index_name} dim={dim} (model={INDEX_DIM})")
    print(f"[create_index] host={host} project={proj} cloud={cloud} region={region}")
    if dim != INDEX_DIM:
        raise ValueError(
            f"Index dim {dim} != model dim {INDEX_DIM}. "
            "Delete the index in the console or change INDEX_DIM/model to match."
        )


def embed_and_upsert_fn(**ctx):
    """
    Embed the lines with Sentence-Transformers and upsert into Pinecone.
    """
    api_key, index_name, *_ = _vars()
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    # sanity check: dimension match
    dim = pc.describe_index(index_name).get("dimension")
    if dim != INDEX_DIM:
        raise ValueError(
            f"Index dim {dim} != model dim {INDEX_DIM}. "
            "Delete the index in console or change INDEX_DIM/model to match."
        )

    # read input file path from XCom (created by prepare_input)
    path = ctx["ti"].xcom_pull(key="input_path")
    print(f"[embed_and_upsert] reading: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    print(f"[embed_and_upsert] {len(lines)} lines")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vectors = model.encode(lines, convert_to_numpy=True)

    items = [
        {"id": f"line-{i}", "values": vec.tolist(), "metadata": {"text": text}}
        for i, (text, vec) in enumerate(zip(lines, vectors))
    ]
    index.upsert(items)
    print(f"[embed_and_upsert] upserted {len(items)} vectors")


def query_index_fn(**_):
    """
    Query the index with a simple semantic question and print top matches.
    """
    api_key, index_name, *_ = _vars()
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    query = "What is Pinecone used for?"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    qvec = model.encode([query])[0].tolist()

    res = index.query(vector=qvec, top_k=3, include_metadata=True)
    print("QUERY:", query)
    for m in res["matches"]:
        txt = (m.get("metadata") or {}).get("text")
        print(f"- id={m['id']} score={m['score']:.4f} text={txt}")


# ---------- DAG ----------
default_args = {"owner": "airflow", "retries": 0}

with DAG(
    dag_id="pinecone_pipeline_demo",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,   # run manually
    catchup=False,
    dagrun_timeout=timedelta(minutes=15),
    tags=["pinecone", "demo", "embeddings"],
) as dag:

    prepare_input = PythonOperator(
        task_id="prepare_input",
        python_callable=prepare_input_fn,
    )

    create_index = PythonOperator(
        task_id="create_index",
        python_callable=create_index_fn,
    )

    embed_and_upsert = PythonOperator(
        task_id="embed_and_upsert",
        python_callable=embed_and_upsert_fn,
    )

    query_index = PythonOperator(
        task_id="query_index",
        python_callable=query_index_fn,
    )

    # pipeline order
    prepare_input >> create_index >> embed_and_upsert >> query_index
