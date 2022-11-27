import pandas as pd

# define a function to remove duplicate tuples from a list
def get_rpc_node_list(model_num):

    # return a dataframe containing all spanIDs and dictionary for all different RPC types
    my_span_list_df = pd.read_csv("data/V" + str(model_num) + "_mySpanDataDF.csv",
                                  encoding='latin-1', sep=',', keep_default_na=False)
    
    my_span_list_df_2 = my_span_list_df.values.tolist()

    # iterate through data frame
    # gather all unique microservice rpc pairs (src, dst)
    rpc_df = pd.DataFrame(columns=['nodeID', 'source', 'destination', 'source_call', 'destination_call'])
    rpc_node_dict = {}

    destinations = list(my_span_list_df['destination'].unique())
    print('Length of my span list',my_span_list_df.shape)
    print('Length of destination',len(destinations))
    #print('Length of Unique Tuples',len(uniqueTuples))
    source = 1
    destination = 2
    rpcCall = 3
    rpcNumber = 4

    
    # iterate through dataframe twice to discover src and dst pairs
    i = 0
    for row in my_span_list_df_2:
        i = i + 1
        #if i%1000==0:
        #    print("\ri:", str(i))
        # find first span of every trace
        # first node of tuple will have a null value for source
        if row[source] == '':
            #print("source is empty i:", str(i))
            null_tuple = (0, row[rpcNumber])
            if null_tuple not in rpc_node_dict.values():
                rpc_node_dict[row[destination]] = null_tuple
                rpc_df = rpc_df.append({
                    'nodeID': str(row[destination]),
                    'source': '0',
                    'destination': str(row[rpcNumber]),
                    'source_call': '',
                    'destination_call': row[rpcCall]},
                                     ignore_index=True)
                continue #This is commented due to missing node pair

        # check for traces with no parent (source) span (see Jaeger and csv file)
        if row[source] not in destinations:
            #print("source is not in destinations i:", str(i))
            start_tuple = (0, row[rpcNumber])

            if start_tuple not in rpc_node_dict.values():
                rpc_node_dict[row[destination]] = start_tuple
                rpc_df = rpc_df.append({
                    'nodeID': str(row[destination]),
                    'source': '0',
                    'destination': str(row[rpcNumber]),
                    'source_call': '',
                    'destination_call': str(row[rpcCall])},
                                    ignore_index=True)
                continue

        #print("just before double iteration i:", str(i))
        #j = 0;
        for col in my_span_list_df_2:
            #if j%10==0:
            #    print('i=',i,'j=',j,row['destination'],col['source'])
            if row[destination] == col[source]:
                row_tuple = (row[rpcNumber], col[rpcNumber])
                #print(row_tuple)
                if row_tuple not in rpc_node_dict.values():
                    rpc_node_dict[col[destination]] = row_tuple
                    rpc_df = rpc_df.append({
                        'nodeID': str(col[destination]),
                        'source': row[rpcNumber],
                        'destination': col[rpcNumber],
                        'source_call': row[rpcCall],
                        'destination_call': col[rpcCall]},
                                         ignore_index=True)
                    continue

    node_ids_list = list(rpc_node_dict.keys())
    node_dim = len(node_ids_list)

    return rpc_df, node_ids_list, node_dim

def step3(model_number):    
    rpcNodeDF, nodeIDsList, n_nodes = get_rpc_node_list(model_number)

    # set the index column to nodeID
    rpcNodeDF = rpcNodeDF.set_index('nodeID')
    rpcNodeDF.to_csv('data/V'+str(model_number)+'_step3_myStaticRPCNodeList.csv',
                     sep=',', encoding='utf-8')

    # append node IDS to model data
    with open("data/V" + str(model_number) + "_rpcNodeInfo.txt", "a") as file:
        file.write("\n\nNumber of different nodes: %s" % n_nodes)
        file.write("\n\nList of RPC nodes:\n")
        for node in nodeIDsList:
            file.write("%s\n" % node)


    # write all node ids to a .txt file
    with open("data/V"+str(model_number)+"_rpcPairNodes.txt", 'w') as f:
        totalLen = len(nodeIDsList)
        i = 1
        for nodeID in nodeIDsList:
            if i==totalLen:
                f.write("%s" % nodeID)
            else:
                f.write("%s," % nodeID)
            i = i + 1
    
    print('rpcPairNodes and myStaticRPCNodeList are created')