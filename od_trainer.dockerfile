#starting from base image
#FROM python:3.8-slim
FROM python:3.10.8-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install dvc 'dvc[gs]'

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .git/ .git/
COPY .dvc/config .dvc/config
COPY data.dvc data.dvc

WORKDIR /
RUN pip install --upgrade pip \
    pip install -r requirements.txt --no-cache-dir

RUN set -e
RUN dvc pull

##name training script as the entry point for docker img
ENTRYPOINT [ "python", "-u" , "src/models/train_model.py"]
