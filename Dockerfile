# One image for both the API and the Streamlit UI (docker-compose picks the command).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /code

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt \
 # Download the tokenizer the text splitter uses at build time, so the
 # container doesn't need internet access at runtime.
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Only what the app needs: the shared research helper and the app itself.
COPY rag_utils.py ./
COPY bbo_attack.py msmarco_queries_valid18.txt msmarco_query_meta.json ./
COPY suffix_attack ./suffix_attack
COPY suffix_attack_results/batch_blockers_distraction_nosuffix_valid18.json ./suffix_attack_results/batch_blockers_distraction_nosuffix_valid18.json
COPY scripts ./scripts
COPY examples ./examples
COPY app ./app
COPY ui ./ui
COPY tests ./tests
COPY evaluation ./evaluation
COPY pytest.ini ./

RUN useradd --create-home appuser && mkdir -p /code/data && chown -R appuser /code
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
