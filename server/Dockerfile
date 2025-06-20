FROM continuumio/miniconda3:25.3.1-1

# Create a new conda environment with python 3.1 and activate it
RUN conda create -n server_conda python=3.11 && \
    conda clean --all -y
RUN echo "source activate server_conda" > ~/.bashrc
ENV PATH=/opt/conda/envs/server_conda/bin:$PATH

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Copy project
COPY . /app
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y libgl1 libegl1
RUN /bin/bash -c "source activate server_conda && \
    conda install -c conda-forge \
    opencv \    
    fastapi \
    insightface \
    -y"
RUN /bin/bash -c "source activate server_conda && \
    conda install -c pytorch -c nvidia -c conda-forge \ 
    pytorch \
    torchvision \ 
    pytorch-cuda=11.8 \
    mkl \
    ultralytics \
    -y"
RUN /bin/bash -c "source activate server_conda && pip install -r requirements.txt"
    
# Expose FastAPI port
EXPOSE 8000
