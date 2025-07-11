FROM python:3.9 as builder

# Install build dependencies
RUN pip install --upgrade pip
RUN pip wheel --no-cache-dir --wheel-dir /wheels opencv-python-headless==4.8.1.78 mediapipe==0.10.7 websockets==11.0.3 numpy==1.24.3

FROM python:3.9-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pre-built wheels and install
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links /wheels opencv-python-headless mediapipe websockets numpy

# Copy application
COPY . .

EXPOSE 8765
ENV PYTHONUNBUFFERED=1
CMD ["python", "websocket_server.py"]
