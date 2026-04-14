FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    unzip \
    libgl1 \
    libglib2.0-0

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy training script
WORKDIR /app
COPY train.py . 
COPY utils.py .

CMD ["python3", "train.py"]
