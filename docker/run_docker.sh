#!/bin/bash
	# --user $(id -u):$(id -g) \
	# --user root \
	--entrypoint="" \
#--mount type=bind,src="/AJAX_STOR/amahmood/",target=/ajax \

docker run \
	--rm \
	-d \
	--init \
	-it \
	--name nvidia-pytorch \
	--ipc=host \
	-p 9009:8888 \
	-e JUPYTER_ENABLE_LAB=yes \
	--gpus device=all \
	--entrypoint="" \
	--mount type=bind,src=/ASD/ahsan_projects/,target=/ahsan_projects \
	--mount type=bind,src="/BEE/Connectome/ABCD/",target=/DATA \
	--mount type=bind,src="/ASD/ahsan_projects/tensorflow_datasets",target=/root/tensorflow_datasets \
	ahsanmah/pytorch_sde:latest \
	bash -c '
	jupyter lab --ip 0.0.0.0 --notebook-dir=/ --no-browser --allow-root
	'

	#  --config=/opt/jupyter/.jupyter/jupyter_notebook_config.py
	# bash -c 'source /etc/bash.bashrc &&
	# echo Starting Jupyter Lab...
	# jupyter lab --notebook-dir=/ --ip 0.0.0.0 --no-browser
	# '
