import pandas as pd
import numpy as np
from datetime import datetime

def convert_func(val):
    val = int(val)
    d_value = datetime.fromtimestamp(float(val / 1000000))
    #d_value = datetime.fromtimestamp(float(val / 1000))
    return d_value


# function used to return elapsed time between first and last time stamps
# in seconds, milliseconds, microseconds
def get_time_data(df):

    # get the first and last timestamps in data frame
    first_format_ts = datetime.strptime(df['rpcTimestamp'][0], '%Y-%m-%d %H:%M:%S.%f')
    last_format_ts = datetime.strptime(df['rpcTimestamp'][len(df)-1], '%Y-%m-%d %H:%M:%S.%f')
    #first_format_ts = datetime.strptime(df['rpcTimestamp'][0], '%Y-%m-%d %H:%M:%S')
    #last_format_ts = datetime.strptime(df['rpcTimestamp'][len(df)-1], '%Y-%m-%d %H:%M:%S')

    # get elapsed time between two timestamps
    time_diff = last_format_ts - first_format_ts

    # get the time difference in seconds
    time_diff_secs = int(time_diff.seconds)

    # get time difference in milliseconds
    time_diff_millis = int((time_diff.seconds * 1000) + (time_diff.microseconds / 1000))

    # get time difference in microseconds
    time_diff_micro = int((time_diff.seconds * 1000000) + time_diff.microseconds)

    return time_diff_secs, time_diff_millis, time_diff_micro


def return_nodes_and_time_intervals(m_num, n_time_interval = 100):

    # read in span data to return time stamped data
    node_span_df = pd.read_csv("data/V" + str(m_num) + "_mySpanDataDF.csv",
                               encoding='latin-1', sep=',', keep_default_na=False)

    # read in time-related data
    time_data_secs, millis_time_data, micro_time_data = get_time_data(node_span_df)

    # return time interval size divisor value
    # get list of 160 timestamped intervals between first and lat timestamp

    # create a number of time windows
    #n_time_interval = 100
    micro_interval_len = round(micro_time_data / n_time_interval)
    milli_interval_len = round(millis_time_data / n_time_interval)

    # fixed periods for input data are in microseconds (3.5 seconds)
    first_ts = node_span_df['rpcStartTime'][0]
    last_ts = node_span_df['rpcStartTime'][len(node_span_df)-1]
    print('first_ts',first_ts)
    print('last_ts',last_ts)

    with open("data/V" + str(m_num) + "_rpcNodeInfo.txt", "a") as file:
        file.write("\n\nTemporal DATA\n")
        file.write("\nFirst timestamp: %s" % str(first_ts))
        file.write("\nFinal timestamp: %s" % str(last_ts))
        file.write("\nTime Difference in seconds: %s" % time_data_secs)
        file.write("\nTime Difference in milliseconds: %s" % millis_time_data)
        file.write("\nTime Difference in microseconds: %s" % micro_time_data)
        file.write("\n\nNo. of time windows: %s" % str(n_time_interval))
        file.write("\nSize of time windows: %s microseconds" % str(micro_interval_len))
        file.write("\nSize of time windows: %s milliseconds" % str(milli_interval_len))

    # get the list of temporal intervals to discover fixed time periods
    time_intervals = list()
    print(range(first_ts, last_ts+1))
    print(micro_interval_len)
    
    
    rnd_ts = 0
    #tmp_ts = first_ts
    #while tmp_ts < last_ts+1:
     #   time_intervals.append(tmp_ts)
      #  rnd_ts = tmp_ts
       # tmp_ts = tmp_ts + micro_interval_len
        
    
    for ts in range(first_ts, last_ts+1):
        #print(ts)
        #if ts==first_ts
            #ts = ts+micro_interval_len-1
        if ts % micro_interval_len == 0:
            #print('adding ts ',ts)
            time_intervals.append(ts)
    #        ts = ts + micro_interval_len # This line is added to increase performance
            rnd_ts = ts

    # (note) append additional timestamp value for any leftover traffic
    if rnd_ts < last_ts:
        rm_val = rnd_ts + micro_interval_len
        time_intervals.append(rm_val)

    return time_intervals


# helper function to compute the traffic of all microservice pair nodes at fixed time periods
def compute_microservice_traffic(model_num,n_time_interval):
    # read in list of timestamp intervals
    print('get time intervals')
    time_stamp_intervals = return_nodes_and_time_intervals(model_num,n_time_interval)
    print('time_stamp_intervals - ',time_stamp_intervals)
    print('length of time stamp intervals - ',len(time_stamp_intervals))
    
    # get list of static nodes to build data frame x-axis
    with open("data/V"+str(model_num)+"_rpcPairNodes.txt", "r") as nodeReader:
        node_ids = nodeReader.read().strip().split(',')

    # read in all occurrences of nodes into new dataframe
    node_instances_df = pd.read_csv('data/V'+str(model_num)+'_myRPCNodeInstancesDF.csv',
                                    encoding='latin-1', sep=',', keep_default_na=False)
    node_instances_df_list = node_instances_df.values.tolist()
    print('length = ',len(node_instances_df_list))

    # build a new data frame using node ids and last df columns
    node_traffic_df = pd.DataFrame(0, index=node_ids, columns=time_stamp_intervals)
    
    #print('node_traffic_df dimension:',node_traffic_df)

    nodeID = 2
    StartTime = 4
    start_time_period = 0
    # compute and increment traffic in all different time periods one by one
    for end_time_interval in time_stamp_intervals:
        #i = 0
        for row in node_instances_df_list:
            #i = i + 1
            node_id = row[nodeID]
            time_value = row[StartTime]
            #if(i%1000==0):
                #print('processing time_interval:',end_time_interval,' with record no - ',i)
                #print(node_id,time_value)
            # check if timestamp is within the outer time interval
            if time_value <= end_time_interval:

                # check if timestamp comes after the inner time interval
                if start_time_period < time_value:

                    # increment instances of nodes by time period
                    node_traffic_df.at[node_id, end_time_interval] += 1
                    #print('increment instances of nodes by time period',node_traffic_df.at[node_id, end_time_interval])
            else:
                start_time_period = end_time_interval
                break
    return node_traffic_df

def step6(model_number,n_time_interval,out_filename):
    print('step 6 started - compute the traffic of all microservice pair nodes at fixed time periods')
    rpcNodeTrafficDF = compute_microservice_traffic(model_number,n_time_interval)
    print('computing microservice traffic for temporal interval is completed')

    # replace columns of milliseconds values with corresponding timestamps
    milli_time_list = list(rpcNodeTrafficDF.columns)
    date_time_list = []

    for ms_time in range(0, len(milli_time_list)):

        # convert millisecond values to timestamps
        ts_value = milli_time_list[ms_time]
        ts_time = convert_func(ts_value)

        # convert timestamps to datetime objects
        date_time_value = np.datetime64(ts_time)
        date_time_list.append(date_time_value)

    # assign the datetime objects to the fixed time periods
    rpcNodeTrafficDF.columns = date_time_list

    rpcNodeTrafficDF.T.to_csv('data/output/ms_traffic_V'+str(model_number)+'.csv',
                            sep=',', encoding='utf-8', index_label='node_ID')
    rpcNodeTrafficDF.T.to_csv(out_filename,
                            sep=',', encoding='utf-8',header=False,index=False)

    print('Successfully created temporal traffic file')