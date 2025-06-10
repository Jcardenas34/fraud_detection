FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.6-py3

# Set the working directory
WORKDIR /app

# Copy code and model
COPY . /app

# Install dependencies (system + Python)
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
RUN pip3 install --upgrade pip

# Necessary for uvicorn
RUN apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN pip3 install -e .


# Expose port for inference API (optional)
EXPOSE 8000

# Command to run your inference script
CMD ["bash", "-c", "scripts/start_fastapi_server.sh "]
