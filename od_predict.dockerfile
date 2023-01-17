# RUNNING od_predict docker image: 
#    docker run --name od_predict --rm \
#       -v %cd%/trained_model.pt:/models/trained_model.pt \  # mount trained model file
#       -v %cd%/data/example_images.npy:/example_images.npy \  # mount data we want to predict on
#        od_predict:latest \
#       ../../models/trained_model.pt \  # argument to script, path relative to script location in container
#       ../../example_images.npy
############


# Starting from base image (generic step 1)
#FROM python:3.8-slim
FROM python:3.10.8-slim

# Install python (generic step 2)
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential parts of application from our computer to container
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

##name training script as the entry point for docker img
ENTRYPOINT [ "python", "-u" , "src/models/predict_model.py"]
#ENTRYPOINT ["python", "-u", "main.py"]

