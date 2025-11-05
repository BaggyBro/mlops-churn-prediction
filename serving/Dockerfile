# Use official Python slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app code
COPY app.py .

# Copy the trained AutoGluon predictor directory (update this path)
COPY autogluon_predictor/ autogluon_predictor/

# Expose port for FastAPI
EXPOSE 8000

# Run server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
