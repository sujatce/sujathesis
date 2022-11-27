import pandas as pd
import numpy as np
import pickle

# build a distance matrix for all RPC nodes
def get_adjacency_matrix(model_num):

    # read from csv file of static nodes (RPC call pairs) into a list
    static_nodes_list = pd.read_csv('data/V'+str(model_num)+'_step3_myStaticRPCNodeList.csv',
                                    encoding='latin-1', sep=',', keep_default_na=False)

    # define the size of adj_matrix using number of static nodes
    l_matrix = len(static_nodes_list)

    # define matrix dimensions = (l_nodes_list x l_nodes_list)
    adj_matrix = np.zeros((l_matrix, l_matrix), dtype=np.float32)
    adj_matrix[:] = np.inf

    for i, row in static_nodes_list.iterrows():

        for j, col in static_nodes_list.iterrows():

            # build the distance matrix
            if row['source'] == col['source'] or row['destination'] == col['destination']:
                adj_matrix[i, j] = 0.5
                adj_matrix[j, i] = 0.5

            if row['source'] == col['destination']:
                adj_matrix[j, i] = 1.0

            if row['destination'] == col['source']:
                adj_matrix[i, j] = 1.0

    #print(adj_matrix)
    #adj_matrix.to_csv('data/V'+str(model_number)+'_adjacency_matrix.csv',
     #                sep=',', encoding='utf-8')
    np.savetxt(f"data/output/adjacency_matrix_V{model_num}.csv", adj_matrix, delimiter=",")
    print('Adjacency Matrix is created')
    return adj_matrix
    
def step4(model_number):
    distanceMatrix = get_adjacency_matrix(model_number)

    # write the distance matrix to pickle file in byte form
    #with open('data/output/adj_mx_V'+str(model_number)+'.pkl', 'wb') as out_File:
    #    pickle.dump(distanceMatrix, out_File, protocol=2)
