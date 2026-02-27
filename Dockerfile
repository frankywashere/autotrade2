FROM python:3.12-slim

# Install system dependencies for C++ extension
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libeigen3-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy everything
COPY . .

# Install Python dependencies (includes C++ extension build via '.')
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace Spaces runs on port 7860
EXPOSE 7860

# Launch Streamlit on the correct port
CMD ["streamlit", "run", "v15/dashboard.py", "--server.port=7860", "--server.address=0.0.0.0"]
