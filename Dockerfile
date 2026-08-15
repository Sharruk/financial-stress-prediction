FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn streamlit plotly

# Copy application
COPY . .

# Expose Streamlit and FastAPI ports
EXPOSE 8501 8000

# By default, start both using a simple script (or we can use docker-compose)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"]
