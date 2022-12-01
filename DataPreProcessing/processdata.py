from step1_format_timestamp import *
from step2_format_trace_spans import step2
from step3_ReturnRPCNodePairsFile import step3
from step4_GenerateAdjacencyMatrix import step4
from step5_RecordNodeInstances import step5
from step6_ReturnRPCTrafficFile import step6

test_number = 50
n_time_interval = 200
#filename = "data/bruteForceAttackData_V"+str(test_number)+".csv"
filename = "data/input/trafficData_V"+str(test_number)+".csv"
out_filename = 'data/V'+str(test_number)+'_step1_TS_trafficData.csv'
out_adj_file = '../suja_stgcn/dataset/ms_traffic_W_'+str(test_number)+'.csv'
out_temporal_file = '../suja_stgcn/dataset/ms_traffic_V_'+str(test_number)+'.csv'

print('Goal: using the input file ',filename,' , prepare a graphical representation of RPC Node Pair calls and its traffic distribution into File1: adjacency weight matrix(spatial data) and File2:traffic data (Temporal Data)')

### STEP 1 - From the trace/span file, just format the timestamp ###
translate_timestamp_data(filename,test_number,out_filename)

### STEP 2 - 
print('Step 2 - Translate all traces and spans into trace,source, destination, rpcCall, rpcNumber, startTime and Timestamp. Also summarize the total event types')
step2(test_number,out_filename)

### STEP 3
print('Step 3 - This step extracts all RPCNodePairs from traffic file.')
step3(test_number)

print('Step 4 - This step generates Weighted Adjacency Matrix from RPC Node pair list, This is Spatial representation of Traffic Data')
step4(test_number,out_adj_file)

print('Step 5 - This step translates traffic file into graphical representation such as each traffic call with its corresponding RPC node pairs')
step5(test_number)

print('Step 6 - This step creates Temporal file for traffic data by translating the traffic timestamps into given intervals and populating the total traffic within each timeslot for each RPC Node Pair-Traffic Node - This is Temporal Traffic file')
step6(test_number,n_time_interval-1,out_temporal_file)
