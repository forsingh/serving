#!/usr/bin/env bash

#this should be run inside docker

/serving/bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=default --model_base_path=/tmp/models &> /tmp/models/model.log &
/serving/bazel-bin/tensorflow_serving/example/flask_client &> /tmp/models/flask.log &
cd ~/jupyter_notebooks
jupyter notebook --ip=0.0.0.0 --NotebookApp.token='' &> /tmp/models/jupyter.log &
#cd /tmp/models

#/bin/bash
jupyter notebook --ip=0.0.0.0 --NotebookApp.token=''