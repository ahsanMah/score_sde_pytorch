FROM gcr.io/kaggle-gpu-images/python

ENV JUPYTER_TOKEN="niral"
ENV PASSWORD=niral

RUN apt-get update --fix-missing
# RUN apt-get install htop tmux screen -y
# RUN curl -fsSL https://deb.nodesource.com/setup_lts.x -o nodesource_setup.sh
# RUN bash nodesource_setup.sh && apt-get install -y nodejs

COPY base_requirements.txt base_requirements.txt
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install -U -r base_requirements.txt
RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# Repo requirements
COPY repo_requirements.txt requirements.txt
RUN pip install -r requirements.txt
