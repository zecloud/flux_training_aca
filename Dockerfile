FROM nvidia/cuda:12.2.2-base-ubuntu22.04 


RUN apt-get update

ENV PYTHON_VERSION=3.10 \  
    DEBIAN_FRONTEND=noninteractive  

RUN apt-get -qq update \  
    && apt-get  -qq  upgrade \
    && apt-get -qq install --no-install-recommends \  
    git \  
    python${PYTHON_VERSION} \  
    python${PYTHON_VERSION}-venv \  
    python3-pip \  
    libopencv-dev \  
    python3-opencv \
    ffmpeg \
    libx264-dev 

# Set up Python aliases  
RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \  
    && ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python \  
    && ln -s -f /usr/bin/pip3 /usr/bin/pip  

WORKDIR /app
ARG CACHEBUST=1
RUN git clone https://github.com/ostris/ai-toolkit.git && \
    cd ai-toolkit && \
    git submodule update --init --recursive

WORKDIR /app/ai-toolkit

#RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install -r requirements.txt
RUN python -m pip install azure-storage-queue


    
