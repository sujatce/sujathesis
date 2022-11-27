from step1_format_timestamp import *
from step2_format_trace_spans import step2
from step3_ReturnRPCNodePairsFile import step3
from step4_GenerateAdjacencyMatrix import step4
from step5_RecordNodeInstances import step5
from step6_ReturnRPCTrafficFile import step6

test_number = 20
n_time_interval = 200
filename = "data/bruteForceAttackData_V"+str(test_number)+".csv"
#filename = "data/trafficData_V"+str(test_number)+".csv"
#out_filename = 'data/V'+str(test_number)+'_step1_TS_trafficData.csv'
print('Goal: using the input file ',filename,' , prepare a graphical representation of RPC Node Pair calls and its traffic distribution into File1: adjacency weight matrix(spatial data) and File2:traffic data (Temporal Data)')
### STEP 1 - From the trace/span file, just format the timestamp ###
#translate_timestamp_data(filename,test_number,out_filename)
### STEP 2 - 
print('Step 2 - Translate all traces and spans into trace,source, destination, rpcCall, rpcNumber, startTime and Timestamp. Also summarize the total event types')
#step2(test_number,out_filename)
### STEP 3
#step3(test_number)
#step4(test_number)
#step5(test_number)
step6(test_number,n_time_interval)
