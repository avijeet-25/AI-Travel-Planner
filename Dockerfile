# Use the Python 3.12 image
FROM python:3.12-slim

WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit uses port 8501 by default
EXPOSE 8501

# Command to run on start
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]