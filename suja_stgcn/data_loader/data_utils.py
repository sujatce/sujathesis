from utils.math_utils import z_score

import numpy as np
import pandas as pd
import os
import csv


def writeToCSV(filename,jCount,iCount,td,n_route_count):
    with open(filename, 'wt') as f:
        writer = csv.writer(f)
        row1 = ["" for x in range(n_route_count+2)]
        row1[0] = 'id1'
        row1[1] = 'id2'
        for indVar in range(n_route_count):
            row1[indVar+2] = str(indVar+1)
        writer.writerow(row1)
        #writer.writerow(('id1', 'id2','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58'))
        for j in range(jCount):
            for i in range(iCount):
                row = np.zeros(n_route_count+2)
                row[0] = str(j + 1)
                row[1] = str(i + 1)
                for k in range(n_route_count):
                    row[k+2] = td[j][i][k][0]
                #row = (
#                    j + 1,
#                    i + 1,
#                    td[j][i][0][0],
#                    td[j][i][1][0],
#                    td[j][i][2][0],
#                    td[j][i][3][0],
#                    td[j][i][4][0],
#                    td[j][i][5][0],
#                    td[j][i][6][0],
#                    td[j][i][7][0],
#                    td[j][i][8][0],
#                    td[j][i][9][0],
#                    td[j][i][10][0],
#                    td[j][i][11][0],
#                    td[j][i][12][0],
#                    td[j][i][13][0],
#                    td[j][i][14][0],
#                    td[j][i][15][0],
#                    td[j][i][16][0],
#                    td[j][i][17][0],
#                    td[j][i][18][0],
#                    td[j][i][19][0],
#                    td[j][i][20][0],
#                    td[j][i][21][0],
#                    td[j][i][22][0],
#                    td[j][i][23][0],
#                    td[j][i][24][0],
#                    td[j][i][25][0],
#                    td[j][i][26][0],
#                    td[j][i][27][0],
#                    td[j][i][28][0],
#                    td[j][i][29][0],
#                    td[j][i][30][0],
#                    td[j][i][31][0],
#                    td[j][i][32][0],
#                    td[j][i][33][0],
#                    td[j][i][34][0],
#                    td[j][i][35][0],
#                    td[j][i][36][0],
#                    td[j][i][37][0],
#                    td[j][i][38][0],
#                    td[j][i][39][0],
#                    td[j][i][40][0],
#                    td[j][i][41][0],
#                    td[j][i][42][0],
#                    td[j][i][43][0],
#                    td[j][i][44][0],
#                    td[j][i][45][0],
#                    td[j][i][46][0],
#                    td[j][i][47][0],
#                    td[j][i][48][0],
#                    td[j][i][49][0],
#                    td[j][i][50][0],
#                    td[j][i][51][0],
#                    td[j][i][52][0],
#                    td[j][i][53][0],
#                    td[j][i][54][0],
#                    td[j][i][55][0],
#                    td[j][i][56][0],
#                    td[j][i][57][0],
#                    td[j][i][58][0]
#                    )
                writer.writerow(row)


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    '''
    Generate data in the form of standard sequence unit.
    :param len_seq: int, the length of target date sequence.
    :param data_seq: np.ndarray, source data / time-series.
    :param offset:  int, the starting index of different dataset type.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param n_route: int, the number of routes in the graph.
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :param C_0: int, the size of input channel.
    :return: np.ndarray, [len_seq, n_frame, n_route, C_0].
    '''
    n_slot = day_slot - n_frame + 1
    print('Length of target data sequence, len_seq=',len_seq)
    print('Number of frame within a standard sequence unit, n_frame=',n_frame)
    print('day_slot=',day_slot)
    print('n_slot=day_slot-n_frame+1=',n_slot)
    print('n_route=',n_route)
    print('Starting index of different dataset type, offset=',offset)
    print('C_0: size of the input channel=',C_0)
    
    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    print('tmp_seq.shape=',tmp_seq.shape)
    #print(tmp_seq)
    
    for i in range(len_seq): #sequence of dataset length (train = 34, validation = 5, test = 5)
        for j in range(n_slot): #268 out of 288 slots are considered per day -- 20 slots are missed out every day_slot, hence 34+5+5 = 44 days, out of which 44 day's last 20 intervals = 880 records are omitted - Not sure if this is intentional
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            #if j==n_slot-1:
            #print('Pick records from ',sta+1,' to ',end)
            #print('put them into ',i*n_slot+j+1)
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])
            #if i==0 and j==0:
            #    print(data_seq[sta:end,:])
            #print(tmp_seq[i * n_slot + j, :, :, :])
    
    return tmp_seq


def data_gen(file_path, data_config, n_route, n_frame=5, day_slot=20): #day_slot 288 is changed here,n_frame = 5
    '''
    Source file load and dataset generation.
    :param file_path: str, the file path of data source.
    :param data_config: tuple, the configs of dataset in train, validation, test.
    :param n_route: int, the number of routes in the graph.
    :param n_frame: int, the number of frame within a standard sequence unit,
                         which contains n_his = 12 and n_pred = 9 (3 /15 min, 6 /30 min & 9 /45 min).
    :param day_slot: int, the number of time slots per day, controlled by the time window (5 min as default).
    :return: dict, dataset that contains training, validation and test with stats.
    '''
    n_train, n_val, n_test = data_config
    # generate training, validation and test data
    try:
        data_seq = pd.read_csv(file_path, header=None).values
    except FileNotFoundError:
        print(f'ERROR: input file was not found in {file_path}.')

    print(data_seq.shape)
    #writeToCSV('data_seq_orig.csv',data_seq,
    seq_train = seq_gen(n_train, data_seq, 0, n_frame, n_route, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_frame, n_route, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_frame, n_route, day_slot)

    # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}
    print(x_stats)

    #print(seq_train.shape);
    #writeToCSV('seq_train.csv',9112,21,seq_train);
    #writeToCSV('seq_val.csv',1340,21,seq_val);
    #writeToCSV('seq_test.csv',1340,21,seq_test);
    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])

    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]

def writeToCSV3Dim(filename,jCount,iCount,td,n_route_count):
    with open(filename, 'wt') as f:
        writer = csv.writer(f)
        row1 = ["" for x in range(n_route_count+2)]
        row1[0] = 'id1'
        row1[1] = 'id2'
        for indVar in range(n_route_count):
            row1[indVar+2] = str(indVar+1)
        #writer.writerow(('id1', 'id2','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59'))
        for j in range(jCount):
            for i in range(iCount):
                row = np.zeros(n_route_count+2)
                row[0] = str(j + 1)
                row[1] = str(i + 1)
                for k in range(n_route_count):
                    row[k+2] = td[j][k][i]
                #row = (
#                    j + 1,
#                    i + 1,
#                    td[j][0][i],
#                    td[j][1][i],
#                    td[j][2][i],
#                    td[j][3][i],
#                    td[j][4][i],
#                    td[j][5][i],
#                    td[j][6][i],
#                    td[j][7][i],
#                    td[j][8][i],
#                    td[j][9][i],
#                    td[j][10][i],
#                    td[j][11][i],
#                    td[j][12][i],
#                    td[j][13][i],
#                    td[j][14][i],
#                    td[j][15][i],
#                    td[j][16][i],
#                    td[j][17][i],
#                    td[j][18][i],
#                    td[j][19][i],
#                    td[j][20][i],
#                    td[j][21][i],
#                    td[j][22][i],
#                    td[j][23][i],
#                    td[j][24][i],
#                    td[j][25][i],
#                    td[j][26][i],
#                    td[j][27][i],
#                    td[j][28][i],
#                    td[j][29][i],
#                    td[j][30][i],
#                    td[j][31][i],
#                    td[j][32][i],
#                    td[j][33][i],
#                    td[j][34][i],
#                    td[j][35][i],
#                    td[j][36][i],
#                    td[j][37][i],
#                    td[j][38][i],
#                    td[j][39][i],
#                    td[j][40][i],
#                    td[j][41][i],
#                    td[j][42][i],
#                    td[j][43][i],
#                    td[j][44][i],
#                    td[j][45][i],
#                    td[j][46][i],
#                    td[j][47][i],
#                    td[j][48][i],
#                    td[j][49][i],
#                    td[j][50][i],
#                    td[j][51][i],
#                    td[j][52][i],
#                    td[j][53][i],
#                    td[j][54][i],
#                    td[j][55][i],
#                    td[j][56][i],
#                    td[j][57][i],
#                    td[j][58][i]
#                    )
                writer.writerow(row)
