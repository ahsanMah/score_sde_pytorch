FROM nvcr.io/nvidia/pytorch:21.12-py3

ENV JUPYTER_TOKEN="niral"
ENV PASSWORD=niral

ARG USER=docker
ARG UID=1000
ARG GID=1000

RUN useradd --system --create-home --home-dir /home/${USER} \
    --shell /bin/bash --gid root -G sudo --uid ${UID} ${USER}
RUN echo ${USER}:docker | chpasswd
USER ${USER}
WORKDIR /home/${USER}
ENV PATH="${PATH}:/home/${USER}/.local/bin"

# Repo requirements
COPY repo_requirementsv2.txt requirements.txt
RUN python -m pip install --user -r requirements.txt

# RUN apt-get update --fix-missing
# RUN apt install -y htop nodejs

#TODO: Maybe add zsh ricing
