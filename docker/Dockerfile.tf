FROM tensorflow/tensorflow:latest-gpu-jupyter

# ARG USER_ID
# ARG GROUP_ID

# RUN addgroup --gid $GROUP_ID user
# RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
# USER user
ENV JUPYTER_TOKEN="niral"
ENV PASSWORD=niral

RUN apt-get update --fix-missing
RUN apt-get install htop tmux screen -y
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x -o nodesource_setup.sh
RUN bash nodesource_setup.sh && apt-get install -y nodejs

COPY base_requirements.txt base_requirements.txt
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install -U -r base_requirements.txt
# RUN pip install -U tensorflow_addons tensorflow_probability tensorflow_datasets
# RUN pip install -U jupyterlab seaborn pandas numba tqdm pyyaml scipy scikit-learn scipy
# RUN pip install -U scikit-image plotly

# Repo requirements
COPY repo_requirements.txt requirements.txt
RUN pip install -r requirements.txt