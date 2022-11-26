import argparse
import tensorflow as tf
import yaml

from lib.utils import load_graph_data
from model.dcrnn_supervisor import DCRNNSupervisor


def train_model(params):
    with open(params.config_filename) as f:

        supervisor_config = yaml.safe_load(f)

        # loading adjacency matrix from pickle file
        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        adj_matrix = load_graph_data(graph_pkl_filename)

        tf_config = tf.ConfigProto()
        if params.use_cpu_only:
            tf_config = tf.ConfigProto(device_count={'GPU': 0})
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as session:
            supervisor = DCRNNSupervisor(adj_mx=adj_matrix, **supervisor_config)

            supervisor.train(sess=session)


parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default="data/DSB_Data/exp_models/dcrnn_DSB.yaml", type=str,
                    help='Configuration filename for restoring the model.')
parser.add_argument('--use_cpu_only', default=True, type=bool, help='Set to true to only use cpu.')
args = parser.parse_args()

# train the filename contained in the params
train_model(args)
