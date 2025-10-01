FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -U pip && pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/Home.py", "--server.headless=true", "--server.port=8501", "--server.address=0.0.0.0"]
