# Build the Server
FROM python:3.13.1-slim AS server_builder
WORKDIR /home
RUN echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes
RUN apt-get update && apt-get install make \
    apt-utils \
    lsb-release \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    sqlite3 \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libsqlite3-dev \
    unixodbc-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libreadline-dev \
    libffi-dev \
    zlib1g-dev \
    wget \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libgl1
COPY . /home
RUN python -m venv venv && . venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 80

FROM python:3.13.1-slim
WORKDIR /home
COPY --from=server_builder /home/ /home/

ENV APP_ENV=prod APP_ROUTE_PREFIX=""
RUN echo "APT::Get::Assume-Yes \"true\";" > /etc/apt/apt.conf.d/90assumeyes
RUN apt-get update && apt-get install \
    ca-certificates \
    sqlite3 \
    libsqlite3-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0
RUN . venv/bin/activate && python download_models.py
CMD ["sh", "-c", ". venv/bin/activate && uvicorn \"api:app\" --host 0.0.0.0 --port 80 --log-level info"]
