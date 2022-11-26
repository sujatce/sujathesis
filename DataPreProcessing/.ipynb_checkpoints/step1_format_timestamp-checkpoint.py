from datetime import datetime
import pandas as pd

def convert_func(val):
    val = int(val)
    d_value = datetime.fromtimestamp(float(val / 1000000))
    #d_value = datetime.fromtimestamp(float(val / 1000))
    return d_value

def translate_timestamp_data(filename,test_number,out_filename):
    pd.options.mode.chained_assignment = None  # default='warn'
    my_data_frame = pd.read_csv(filename, encoding='latin-1', sep=',')

    my_time_values = []
    # convert microseconds temporal value to actual timestamp
    for i, row in my_data_frame['startTime'].items():
        time_val = convert_func(row)
        if i==1:
            print(row)
            print(time_val)
        my_time_values.append(time_val)

    # assign timestamp values to new column
    my_data_frame['timeStamp'] = my_time_values

    # write updated dataframe to new csv file
    my_data_frame.to_csv(out_filename, sep=',', encoding='utf-8', index=False)
    print('translated timestamp into visible format in CSV file - ',out_filename)

#translate_timestamp_data()
