FROM avloss/tf-serving-docker-base:latest

MAINTAINER Anton Loss @avloss

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/root/.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/root/.bazelrc
ENV BAZELRC /root/.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.4.2
WORKDIR /

#https://github.com/tensorflow/tensorflow/issues/7048
RUN update-ca-certificates -f

RUN mkdir /bazel && \
    cd /bazel && \
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE.txt && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

#maintainer @avloss

RUN mkdir /serving

COPY tensorflow /serving/tensorflow
COPY tensorflow_serving /serving/tensorflow_serving
COPY tf_models /serving/tf_models
COPY tools /serving/tools
COPY WORKSPACE /serving/WORKSPACE

RUN cd /serving/tensorflow && \
    yes "" | ./configure

## limited resources to build locally on my mac!
#RUN cd /serving/ && \
#    bazel build -c opt --local_resources 2048,.5,1.0 tensorflow_serving/...

## unlimited resources to build on the server (dockerhub)
RUN cd /serving/ && \
    bazel build tensorflow_serving/...

RUN pip install tensorflow

RUN mkdir /tmp/models && \
    mkdir /root/jupyter_notebooks

COPY example_jupyter/setup_flask_client.sh /root/setup_flask_client.sh
COPY example_jupyter/test_tf_serving_flask.ipynb /root/jupyter_notebooks/

EXPOSE 8888 9000 5000
CMD ["bash", "/root/setup_flask_client.sh"]