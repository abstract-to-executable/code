
FROM nvidia/cudagl:11.4.2-base
RUN apt-get update
RUN apt-get install -y wget git
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

WORKDIR /workspace
# Setup conda and packages
# only copy scripts and environment.yml so docker does not need to rebuild every time something else changes
COPY environment.yml environment.yml
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
COPY ./nautilus/setup.sh ./nautilus/setup.sh

RUN sh ./nautilus/setup.sh

# COPY . .
# RUN python -m pip install -e .
