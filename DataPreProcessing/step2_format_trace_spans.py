import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


# define function to return DataFrame with appropriate columns, no. of event types,
# data frame with total spans
def read_rpc_data(model_num, inputfilename):
    #  use Pandas library to extract a data frame from csv file
    #traces_df = pd.read_csv("data/trafficData_TS_V" + str(model_num) + ".csv",
    traces_df = pd.read_csv(inputfilename,
                            encoding='latin-1', sep=',', keep_default_na=False)
    # traces_df = pd.read_csv("../data/batchRegistrationAttackData_TS_V" + str(model_num) + ".csv",
    #                         encoding='latin-1', sep=',', keep_default_na=False)
    # traces_df = pd.read_csv("../data/dDoSAttackData_TS_V" + str(model_num) + ".csv",
    #                         encoding='latin-1', sep=',', keep_default_na=False)

    # define a column for event types by combining columns services and operations
    traces_df['RPCSpan'] = traces_df['process.serviceName'] + "+" + traces_df['operationName']

    # get the number of unique event types
    n_rpc_spans = traces_df['RPCSpan'].unique()
    n_event_types = len(n_rpc_spans)

    # generate an encoding dictionary
    r_pc_numbers = range(1, n_event_types + 1)
    my_rpc_dict = dict(zip(n_rpc_spans, r_pc_numbers))

    # map all different events to an event numbers
    traces_df['RPCNumber'] = traces_df['RPCSpan'].map(my_rpc_dict)

    # give column a new name
    traces_df['parentSpanID'] = traces_df['references.0.spanID']

    # filter unnecessary columns
    my_traces_df = traces_df[['traceID', 'spanID', 'RPCSpan', 'RPCNumber', 'parentSpanID', 'timeStamp', 'startTime']]

    # return a data Frame with the appropriate columns
    span_list_df = pd.DataFrame({'trace': my_traces_df['traceID'],
                                 'source': my_traces_df['parentSpanID'],
                                 'destination': my_traces_df['spanID'],
                                 'rpcCall': my_traces_df['RPCSpan'],
                                 'rpcNumber': my_traces_df['RPCNumber'],
                                 'rpcStartTime': my_traces_df['startTime'],
                                 'rpcTimestamp': my_traces_df['timeStamp']})

    return my_traces_df, span_list_df, my_rpc_dict, n_event_types


def output_print_analysis(model_no,myDataFrame,mySpanListDF, encodingRPCDict, nEventTypes):
    # read in the dataframe from csv file
    l_data_frame = len(myDataFrame)

    # group span number by traceIDs into a list
    my_new_data_frame = myDataFrame[['traceID', 'RPCNumber']]
    case_list = my_new_data_frame.groupby('traceID')['RPCNumber'].apply(list)
    data = case_list.tolist()
    n_cases = len(data)

    with open("data/V" + str(model_no) + "_rpcNodeInfo.txt", "w") as file:
        file.write("Test Number: %s" % model_no)
        file.write("\n\nTotal Number of spans: %s" % l_data_frame)
        file.write("\nTotal Number of event types: %s" % nEventTypes)
        file.write("\nTotal number of traces: %s" % n_cases)
    print("Test Number: %s" % model_no)
    print("\n\nTotal Number of spans: %s" % l_data_frame)
    print("\nTotal Number of event types: %s" % nEventTypes)
    print("\nTotal number of traces: %s" % n_cases)
        
    mySpanListDF.to_csv("data/V" + str(model_no) + "_mySpanDataDF.csv", sep=',', encoding='utf-8', index=False)
    print("Span List is stored in ","data/V" + str(model_no) + "_mySpanDataDF.csv")
    
def step2(test_number,inputfilename):
    myDataFrame, mySpanListDF, encodingRPCDict, nEventTypes = read_rpc_data(test_number,inputfilename)
    output_print_analysis(test_number,myDataFrame,mySpanListDF, encodingRPCDict, nEventTypes)