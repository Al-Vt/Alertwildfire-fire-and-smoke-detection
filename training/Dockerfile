FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y \
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

CMD ["python", "train.py"]
