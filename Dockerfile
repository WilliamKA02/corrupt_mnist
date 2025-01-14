FROM python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements_docker.txt requirements_docker.txt
COPY main.py main.py
WORKDIR /
RUN pip install -r requirements_docker.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "main.py"]
