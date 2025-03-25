FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# fix timezone
ENV TZ=Asia/Novosibirsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 wget bzip2 ca-certificates \
    libglib2.0-0 libxrender1 git mercurial subversion && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup conda virtual environment
COPY ./environment.yml /tmp/environment.yml

RUN conda update conda \
    && conda env create --file /tmp/environment.yml

RUN echo "conda activate facade_processing_tools" >> ~/.bashrc
ENV PATH /opt/conda/envs/camera-seg/bin:$PATH
ENV CONDA_DEFAULT_ENV $facade_processing_tools

ARG MY_NAME
ARG MY_UID
ARG MY_GID

RUN groupadd --gid ${MY_GID} ${MY_NAME} && \
    useradd --create-home --shell /bin/bash --uid ${MY_UID} --gid ${MY_GID} \
    --groups sudo -p $(openssl passwd -1 pass) ${MY_NAME}

USER ${MY_NAME}

ENV PATH "${PATH}:/home/${MY_NAME}/.local/bin"
