FROM avloss/tf-serving-jupyter-flask-base:latest

MAINTAINER Anton Loss @avloss

## limited resources to build locally on my mac!
RUN cd /serving/ && \
    bazel build -c opt --local_resources 2048,.5,1.0 tensorflow_serving/...

## unlimited resources to build on the server (dockerhub)
#RUN cd /serving/ && \
#    bazel build tensorflow_serving/model_servers/tensorflow_model_server tensorflow_serving/example/flask_client

RUN pip install tensorflow

RUN mkdir /tmp/models && \
    mkdir /root/jupyter_notebooks

COPY example_jupyter/setup_flask_client.sh /root/setup_flask_client.sh
COPY example_jupyter/test_tf_serving_flask.ipynb /root/jupyter_notebooks/

EXPOSE 8888 9000 5000
CMD ["bash", "/root/setup_flask_client.sh"]