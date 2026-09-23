FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# Railway injects PORT at runtime; must bind 0.0.0.0, not localhost.
# PUBLIC_DEPLOY set here (not in code) - src/app.py checks os.getenv("PUBLIC_DEPLOY")
# to switch the checkpointer from SqliteSaver to MemorySaver and keep the
# saved-paper-library concern (data/library.db) out of the public deploy
# entirely - the local-vs-public split CLAUDE.md documents.
ENV PUBLIC_DEPLOY=1

EXPOSE 8501

CMD ["sh", "-c", "streamlit run src/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true"]
