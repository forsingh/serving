import cPickle as pickle

import numpy as np
import requests
import tensorflow as tf
import numpy as np
from tensorflow.contrib.session_bundle import exporter

VERSION = 44


def create_and_export_model(export_path):
    x = tf.placeholder(tf.int32, shape=[3])
    z = tf.Variable([2])
    y = tf.mul(x, z)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    feed_dict = {x: [3, 4, 5]}
    print(sess.run(y, feed_dict=feed_dict))

    print 'Exporting trained model to', export_path
    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'x': x}),
            'outputs': exporter.generic_signature({'y': y})})
    model_exporter.export(export_path, tf.constant(VERSION), sess)


def test_flask_client():
    URL = "http://localhost:5000/model_prediction"

    a = np.array([1, 2, 3], dtype="int32")
    s = pickle.dumps({"x": a}, protocol=0)

    DATA = {"model_name": "default",
            "input": quote(s)}

    r = requests.post(URL, data=DATA)

    print r.status_code
    print r.text

if __name__=="__main__":
    create_and_export_model("/tmp/models")
    test_flask_client()