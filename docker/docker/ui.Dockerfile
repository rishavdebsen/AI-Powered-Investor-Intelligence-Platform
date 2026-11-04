FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .
EXPOSE 8501
CMD ["streamlit", "run", "src/investor_intel/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
