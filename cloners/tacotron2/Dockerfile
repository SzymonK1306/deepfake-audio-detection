FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=/opt/conda/bin:$PATH

# Install dependencies
RUN apt-get update --fix-missing && apt-get install -y \
    wget \
    git \
    vim \
    ffmpeg \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh \
    && /opt/conda/bin/conda clean --all -f -y

# Create and activate conda environment
RUN /opt/conda/bin/conda create --name taco2 python=3.6.8 \
    && echo "source activate taco2" > ~/.bashrc

# Install Python dependencies
RUN /opt/conda/bin/conda run -n taco2 pip install \
    matplotlib \
    tensorflow==1.15.2 \
    numpy==1.19.4 \
    inflect==0.2.5 \
    librosa==0.6.0 \
    scipy==1.5.4 \
    Unidecode==1.0.22 \
    pillow \
    torch==1.10.2+cu113 torchaudio==0.10.2+cu113 torchvision==0.11.3+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113 \
    numba==0.48.0 \
    future \
    resampy==0.3.1 \
    pydub

# Clone and configure Apex
RUN git clone https://github.com/NVIDIA/apex.git /workspace/apex
WORKDIR /workspace/apex
RUN /opt/conda/bin/conda run -n taco2 pip install -r requirements.txt \
    && /opt/conda/bin/conda run -n taco2 pip install -v --disable-pip-version-check \ 
       --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" .

# Verify CUDA installation
RUN nvcc --version \
    && /opt/conda/bin/conda run -n taco2 python -c "import torch; print(torch.cuda.is_available())"

# Set working directory to tacotron2
RUN mkdir -p /workspace/tacotron2
WORKDIR /workspace/tacotron2

# Activate conda
CMD ["bash", "-c", "source activate taco2"]