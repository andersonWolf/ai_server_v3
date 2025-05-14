# Use slim Python image
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the correct port (8888 as specified in app.py)
EXPOSE 8888

# Run the application using uvicorn
CMD ["uvicorn", "main_v3_header:app", "--host", "0.0.0.0", "--port", "8888"]
