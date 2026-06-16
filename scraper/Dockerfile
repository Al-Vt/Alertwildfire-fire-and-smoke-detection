# On part d'une version Python officielle
FROM python:3.10-slim

# libpq-dev and gcc are used for psycopg2
# wget is used to download with Chrome
# gnup is used for security
# The last line deletes the cache after installation 
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    unzip \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*  

# Chrome installation required for Selenium
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update && apt-get install -y google-chrome-stable

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# PYTHONUNBUFFERED allows you to display logs in real time
ENV PYTHONUNBUFFERED=1  
CMD ["python", "scraper.py"]