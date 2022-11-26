import pandas as pd
import numpy as np
import argparse
import os
import yaml

from a0Functions import get_model_number


# define the function to read in args data frame
def generate_train_val_testing_data(params, model_num):

    # read in h5 file with the RPC node pair traffic as a new data frame
    traffic_df = pd.read_hdf(params.traffic_df_filename)

    # define data array for input and output
    num_nodes, num_time_intervals = traffic_df.shape

    timestamps = traffic_df.columns

    # expand and transpose the dimensions of data frame to match the following shape
    # (num_time_intervals, num_nodes, 1)
    my_expand_traffic = np.expand_dims(traffic_df.values, axis=-1)

    # transposing data (0, 1, 2) = (1, 0, 2)
    my_data = my_expand_traffic.transpose((1, 0, 2))

    # to calculate the predictions
    # define offsets for 1 time step to predict future traffic in subsequent time window
    x_offsets = np.sort(np.arange(0, 1, 1))
    y_offsets = np.sort(np.arange(1, 2, 1))

    # declare x, and y arrays
    x, y = [], []

    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_time_intervals - abs(max(y_offsets)))

    # iterate through my_data index by index to calculate
    for t in range(min_t, max_t):
        x_t = my_data[t + x_offsets, ...]
        y_t = my_data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    # Write the data into npz files
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.15)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train

    # define training data
    x_train, y_train = x[:num_train], y[:num_train]

    # validation
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )

    x_value = x_offsets.reshape(list(x_offsets.shape) + [1])

    # define testing data
    x_test, y_test = x[-num_test:], y[-num_test:]

    # write out timestamps
    test_timestamps = list(timestamps[-num_test:])

    # write the testing timestamps to print out
    with open('../data/testing_Timestamps/test_timestamps_V' + str(model_num) + '.txt', 'w') as file:
        for t_stamp in test_timestamps:
            if test_timestamps[-1] != t_stamp:
                file.write("%s," % t_stamp)
            else:
                file.write(str(t_stamp))

    # declare training, validating and testing data
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        # print(cat)
        # print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(args.output_dir, "%s_V%s.npz" % (cat, str(model_num))),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
        #np.savetxt("%s_V%s.csv" % (cat, str(model_num)), adj_matrix, delimiter=",")

    return num_nodes


model_number = get_model_number()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir", type=str, default="../data/DSB_Data/files", help="Output directory."
)

# use the .h5 file as input traffic file name
parser.add_argument(
    "--traffic_df_filename",
    type=str,
    default='../data/DSB_Data/myRPCNodeTraffic_V'+str(model_number)+'.h5',
)
args = parser.parse_args()

# read in data frame from .h5, generate data sets and update .yaml file
n_nodes = generate_train_val_testing_data(args, model_number)

# read and load the .yaml file
f_name = '../data/DSB_Data/model/dcrnn_DSB.yaml'
stream = open(f_name, 'r')
y_Data = yaml.load(stream, Loader=yaml.FullLoader)

# write and update the .yaml file before training the model
y_Data['model_number'] = model_number
y_Data['model']['horizon'] = 1
y_Data['model']['num_nodes'] = n_nodes
y_Data['data']['graph_pkl_filename'] = '../data/DSB_Data/graph/adj_mx_V'+str(model_number)+'.pkl'

with open(f_name, 'w') as f:
    f.write(yaml.dump(y_Data))
