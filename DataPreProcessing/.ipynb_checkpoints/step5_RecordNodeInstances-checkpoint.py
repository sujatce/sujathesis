import pandas as pd

# write a function to read the node list file and record every RPC pair node instance
def return_node_instance_data(model_num):

    # read from csv file of static nodes (RPC call pairs) into a list
    static_nodes_list = pd.read_csv('data/V'+str(model_num)+'_step3_myStaticRPCNodeList.csv',
                                    encoding='latin-1', sep=',', keep_default_na=False)

    # create a dictionary from data frame with keys for the static values
    nodes_dict = {}
    for i, span in static_nodes_list.iterrows():
        my_tuple = (span['source'], span['destination'])
        nodes_dict[span['nodeID']] = my_tuple

    # return a dataframe containing all rpc spanIDs and node pairs
    span_list_df = pd.read_csv("data/V" + str(model_num) + "_mySpanDataDF.csv",
                               encoding='latin-1', sep=',', keep_default_na=False)

    node_values_df = pd.DataFrame(columns=['traceID', 'nodeID', 'node_pair', 'StartTime', 'Timestamp'])

    destinations = list(span_list_df['destination'].unique())

    # iterate through data frame twice to discover nodes
    for i, row in span_list_df.iterrows():

        # declare initial zero value
        null_num = 0
        row_num = row['rpcNumber']

        #print("i: ", str(i))

        # look for microservice calls at the start of a trace with no source values
        if row['source'] == '':
            start_tuple = (null_num, row_num)
            key = list(nodes_dict.keys())[list(nodes_dict.values()).index(start_tuple)]
            # append node instance to data frame
            node_values_df = node_values_df.append({'traceID': row['trace'],
                                                    'nodeID': key,
                                                    'node_pair': start_tuple,
                                                    'StartTime': row['rpcStartTime'],
                                                    'Timestamp': row['rpcTimestamp']},
                                                   ignore_index=True)

        # look for microservice spans with no listed parent spanID
        elif row['source'] != '' and row['source'] not in destinations:
            branch_tuple = (null_num, row_num)
            key = list(nodes_dict.keys())[list(nodes_dict.values()).index(branch_tuple)]

            # append node instance to data frame
            node_values_df = node_values_df.append({'traceID': row['trace'],
                                                    'nodeID': key,
                                                    'node_pair': branch_tuple,
                                                    'StartTime': row['rpcStartTime'],
                                                    'Timestamp': row['rpcTimestamp']},
                                                   ignore_index=True)

        for j, col in span_list_df.iterrows():
            #print('i=',i,'j=',j,row['destination'],col['source'])
            col_num = col['rpcNumber']

            # check if row's destination matches col's source
            if row['destination'] == col['source']:
                pair_value = (row_num, col_num)
                #print('here is where it goes error',pair_value)
                #print('i=',i,'j=',j)
                #print('nodes dict',nodes_dict)
                key = list(nodes_dict.keys())[list(nodes_dict.values()).index(pair_value)]

                # append node instance to data frame
                node_values_df = node_values_df.append({'traceID': col['trace'],
                                                        'nodeID': key,
                                                        'node_pair': pair_value,
                                                        'StartTime': col['rpcStartTime'],
                                                        'Timestamp': col['rpcTimestamp']},
                                                       ignore_index=True)
    return node_values_df

def step5(model_number):
    myNodeValuesDF = return_node_instance_data(model_number)

    # sort values by time stamp
    myNodeValuesDF = myNodeValuesDF.sort_values('Timestamp')
    myNodeValuesDF.to_csv('data/V'+str(model_number)+'_myRPCNodeInstancesDF.csv',
                          sep=',', encoding='utf-8', index_label='Node')
    
    print('Step 5 Completed - ')
