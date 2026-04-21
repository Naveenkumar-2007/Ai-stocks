FROM apache/airflow:2.10.3-python3.12

USER airflow

# Install project runtime deps inside Airflow image, but keep base Airflow package from the parent image.
COPY requirements.txt /opt/airflow/backend/requirements.txt
RUN bash -c "grep -Ev '^(apache-airflow==|faiss-cpu==|langchain|langchain-community|langchain-groq|groq|sentence-transformers==)' /opt/airflow/backend/requirements.txt > /tmp/requirements-airflow-runtime.txt && pip install --no-cache-dir -r /tmp/requirements-airflow-runtime.txt"
