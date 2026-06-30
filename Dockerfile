# Use a slim Python image to keep the size small
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (required for some Python packages like CrewAI or compilation tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit application
# We listen on 0.0.0.0 so it is accessible outside the container
CMD ["streamlit", "run", "app2.py", "--server.port=8501", "--server.address=0.0.0.0"]
