import os
import csv
import sys
from data_loader.data_utils import writeToCSV

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import join as pjoin

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.Session(config=config)

from utils.math_graph import *
from data_loader.data_utils import *
from models.trainer import model_train
from models.tester import model_test

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_route', type=int, default=63)
parser.add_argument('--n_his', type=int, default=12) #previous = 12
parser.add_argument('--n_pred', type=int, default=9) #previous = 9
day_slot = 25
n_train, n_val, n_test = 2, 1, 1
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--ks', type=int, default=3)
parser.add_argument('--kt', type=int, default=3)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--opt', type=str, default='RMSProp')
parser.add_argument('--graph', type=str, default='default')
parser.add_argument('--inf_mode', type=str, default='sep') #merge

args = parser.parse_args()
print(f'Training configs: {args}')

n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
Ks, Kt = args.ks, args.kt
# blocks: settings of channel size in st_conv_blocks / bottleneck design
blocks = [[1, 32, 64], [64, 32, 128]]

# Load wighted adjacency matrix W
if args.graph == 'default':
    W = weight_matrix(pjoin('./dataset', f'ms_traffic_W_{n}.csv'))
else:
    # load customized graph weight matrix
    W = weight_matrix(pjoin('./dataset', args.graph))
print(W.shape)
#np.savetxt("ms_traffic_Weight_matrix.csv", W, delimiter=",")

# Calculate graph kernel
L = scaled_laplacian(W)
#print('scaled_laplacian')
#np.savetxt("scaled_laplacian.csv", L, delimiter=",")
print(L.shape)
# Alternative approximation method: 1st approx - first_approx(W, n).
Lk = cheb_poly_approx(L, Ks, n)
np.savetxt("cheb_poly.csv", Lk, delimiter=",")
#print(Lk.shape)
tf.add_to_collection(name='graph_kernel', value=tf.cast(tf.constant(Lk), tf.float32))

# Data Preprocessing
data_file = f'ms_traffic_V_{n}.csv'
#n_train, n_val, n_test = 34, 5, 5

input_data = data_gen(pjoin('./dataset', data_file), (n_train, n_val, n_test), n, n_his + n_pred,day_slot)
print('train.shape=',input_data.get_data('train').shape)
print('val.shape=',input_data.get_data('val').shape)
print('test.shape=',input_data.get_data('test').shape)
writeToCSV('input_data_train.csv',input_data.get_data('train').shape[0],input_data.get_data('train').shape[1],input_data.get_data('train'))
writeToCSV('input_data_val.csv',input_data.get_data('val').shape[0],input_data.get_data('val').shape[1],input_data.get_data('val'))
writeToCSV('input_data_test.csv',input_data.get_data('test').shape[0],input_data.get_data('test').shape[1],input_data.get_data('test'))
print(f'>> Loading dataset with Mean: {input_data.mean:.2f}, STD: {input_data.std:.2f}')

if __name__ == '__main__':
    model_train(input_data, blocks, args)
    model_test(input_data, input_data.get_len('test'), n_his, n_pred, args.inf_mode)
