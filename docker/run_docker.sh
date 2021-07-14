#!/bin/bash

# -u 298493:1001 \ -u $(id -u):$(id -g) 
# --group-add users

##### UNCOMMENT THIS FOR FIRST RUN ######
docker build ./ -t ahsanmah/pytorch_sde

docker run \
	-d \
	--rm \
	-it \
	--name ahsan-pytorch \
	-p 9009:8888 \
	-e JUPYTER_TOKEN="niral" \
	-e PASSWORD=niral \
	--gpus device=all \
	--entrypoint="" \
	--mount type=bind,src=/ASD/ahsan_projects/,target=/home \
	--mount type=bind,src="/BEE/Connectome/ABCD/",target=/DATA \
	--mount type=bind,src="/ASD/ahsan_projects/tensorflow_datasets",target=/root/tensorflow_datasets \
	ahsanmah/pytorch_sde \
	bash -c 'source /etc/bash.bashrc &&
	echo Starting Jupyter Lab...
	jupyter lab --notebook-dir=/ --ip 0.0.0.0 --no-browser --allow-root
	'