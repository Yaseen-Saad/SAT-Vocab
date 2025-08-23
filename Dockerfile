FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create directories for data and feedback storage
RUN mkdir -p data/processed feedback_data

# Copy startup script
COPY start.py ./
COPY start.sh ./
RUN chmod +x start.sh

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose port (Railway will set PORT environment variable)
EXPOSE 8000

# Run the application using Python startup script
CMD ["python", "start.py"]