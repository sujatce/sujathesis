from data_loader.data_utils import gen_batch
from data_loader.data_utils import writeToCSV
from data_loader.data_utils import writeToCSV3Dim
from utils.math_utils import evaluation
from utils.math_utils import loss_difference
from os.path import join as pjoin

import tensorflow as tf
import numpy as np
import time
import pandas as pd


def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):
    '''
    Multi_prediction function.
    :param sess: tf.Session().
    :param y_pred: placeholder.
    :param seq: np.ndarray, [len_seq, n_frame, n_route, C_0].
    :param batch_size: int, the size of batch.
    :param n_his: int, size of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param dynamic_batch: bool, whether changes the batch size in the last one if its length is less than the default.
    :return y_ : tensor, 'sep' [len_inputs, n_route, 1]; 'merge' [step_idx, len_inputs, n_route, 1].
            len_ : int, the length of prediction.
    '''
    #print('Multi_Pred Method invoked')
    #print(y_pred) #Tensor("strided_slice_4:0", shape=(?, 228, 1), dtype=float32)
    #print('batch_size = ',batch_size,' ; len(seq) = ',len(seq))
    #print(seq) # This is the input data for which prediction is going to be done
    pred_list = []
    count = 0
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        #print('Loop begin')
        #print('gen_batch i=',i.shape) #This is same as seq, probably len(seq) is smaller than batch_size
        #print(i)
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        #print('test_seq=',test_seq.shape)
        count = count + 1
        #writeToCSV(f'test_seq_{count}.csv',1340,13,test_seq)
        #print(test_seq)
        step_list = []
        #print('Before for loop')
        for j in range(n_pred):
            #print('Inside for loop before sess run-',j)
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            #print('Inside for loop after sess run')
            if isinstance(pred, list):
                pred = np.array(pred[0])
            #print(pred.shape)
            #writeToCSV('pred_1.csv',1340,1,pred)
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            #print('0 to ',n_his-1,' records are replaced by 1 to ',n_his,' records ; basically first record is deleted and predicted record is appended at last but first, last record remains the same')
            #print(pred)
            #print(test_seq.shape)
            #writeToCSV(f'test_seq_{count}_{j}.csv',1340,13,test_seq)
            step_list.append(pred)
        pred_list.append(step_list)
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    #print('End for loop and shape of pred_array as follows')
    pred_array = np.concatenate(pred_list, axis=1)
    #print('Predicted Array shape',pred_array.shape)
    #writeToCSV('pred_array.csv',9,1340,pred_array)
    return pred_array[step_idx], pred_array.shape[1]


def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val):
    '''
    Model inference function.
    :param sess: tf.Session().
    :param pred: placeholder.
    :param inputs: instance of class Dataset, data source for inference.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param step_idx: int or list, index for prediction slice.
    :param min_va_val: np.ndarray, metric values on validation set.
    :param min_val: np.ndarray, metric values on test set.
    '''
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    if n_his + n_pred > x_val.shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')

    #print('model_inference method - Now take x_val data, using which predict y_val')
    y_val, len_val = multi_pred(sess, pred, x_val, batch_size, n_his, n_pred, step_idx)
    #print('Found y_val, now compare x_val with y_val to evaluate how much they match')
    evl_val = evaluation(x_val[0:len_val, step_idx + n_his, :, :], y_val, x_stats)
    #print('evaluataed value=',evl_val,'compare that with min_va_val=',min_va_val)
    
    # chks: indicator that reflects the relationship of values between evl_val and min_va_val.
    chks = evl_val < min_va_val
    #print('chks: indicator that reflects the relationship of values between evl_val and min_va_val = ',chks)
    betterPerformance = False
    # update the metric on test set, if model's performance got improved on the validation.
    if sum(chks):
        betterPerformance = True
        #print('since validation is successful, ie, evaluation values are lesser than minimum validation value, now use the x_test data and do the prediction => y_pred, also make min_va_val as newly found minimum validation value')
        min_va_val[chks] = evl_val[chks]
        y_pred, len_pred = multi_pred(sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        #print('Found y_pred using x_test data, now compare x_test with y_pred')
        evl_pred = evaluation(x_test[0:len_pred, step_idx + n_his, :, :], y_pred, x_stats)
        #print('newly evaluated value is now min_val; min_val = evl_pred(x_test vs y_pred) = ',evl_pred)
        min_val = evl_pred
    return min_va_val, min_val, betterPerformance


def model_test(inputs, batch_size, n_his, n_pred, inf_mode, n, model_num,load_path='./output/models/'):
    '''
    Load and test saved model from the checkpoint.
    :param inputs: instance of class Dataset, data source for test.
    :param batch_size: int, the size of batch.
    :param n_his: int, the length of historical records for training.
    :param n_pred: int, the length of prediction.
    :param inf_mode: str, test mode - 'merge / multi-step test' or 'separate / single-step test'.
    :param load_path: str, the path of loaded model.
    '''
    start_time = time.time()
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path
    print('For model_test, first load the model(*.meta) from ',model_path)

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        print('load the latest checkpoint file')
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        print('Load the y_pred (which is predicted out x_test) from saved model')
        pred = test_graph.get_collection('y_pred')
        print(pred);

        #print('Identify step index from inf_mode (sep or merge)')
        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')

        #print('step_idx')
        #print(step_idx)
        #print('Load the x_test data')
        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()
        #print('x_test dimension',x_test.shape)
        #writeToCSV('x_test.csv',1340,21,x_test)
        #print(x_stats);
        #print('From x_test data, predict y_test data using the test session from saved model')

        y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred, step_idx)
        #print('len_test',len_test)
        #writeToCSV('y_test_extracted_step_idx.csv',3,1340,y_test)
        #n_his = 0;
        #print('y_test.shape = ',y_test.shape) #ex (3, 1340, 228, 1)
        #print('len_test = ',len_test) #ex 1340
        #print('step_idx = ',step_idx) #ex [2 5 8]
        #print('n_his = ',n_his) #ex 12
        #print('step_idx+n_his=',step_idx + n_his) #ex [14 17 20]
         #Take 14th, 17th and 20th (zero based index) records from x_test
        #print('extract x_test for its step_idx records only. its shape is ',x_test[0:len_test, step_idx + n_his, :, :].shape) #ex (1340, 3, 228, 1)
        #writeToCSV('x_test_extracted_step_idx.csv',1340,3,x_test[0:len_test, step_idx + n_his, :, :])
        #print('len(y_test.shape)=',len(y_test.shape)) #ex 4
        #print('Evaluate x_test with y_test for given step index')
        
        x_test_orig = x_test[0:len_test, step_idx + n_his, :, :]
        y_pred_using_train = y_test
        
        loss_diff = loss_difference(x_test_orig,y_pred_using_train,x_stats)
        threshold = (x_stats['mean'] + 2 * x_stats['std']) * 2
        
        print('loss_diff.shape=',loss_diff.shape)
        writeToCSV3Dim('loss_diff_1.csv',loss_diff.shape[0],loss_diff.shape[2],loss_diff,n)
        print('Prediction error Threshold = ', threshold)
        
        
        ms_node_pairs_df = pd.read_csv('../DataPreProcessing/data/V'+str(model_num)+'_step3_myStaticRPCNodeList.csv',
                                    encoding='latin-1', sep=',', keep_default_na=False)
        ms_node_pairs = ms_node_pairs_df.values.tolist()
        print(ms_node_pairs_df.shape)
        #print(ms_node_pairs)
        #print(ms_node_pairs[0][0])
        
                #Detecting Anomoly traffic nodes
        for i in range(loss_diff.shape[0]):
            anomoly_df = pd.DataFrame(columns=[#'id',
                'source', 'destination', 'threshold', 'prediction_error'])
            count = 0
            for j in range(loss_diff.shape[1]):
                #if j < 2:
#                    continue
                for k in range(loss_diff.shape[2]):
                    if loss_diff[i][j][k] > threshold:
                        #print('i=',i,'j=',j,'k=',k)
                        count = count+1
                        print("Anomoly detected at microservices node pair ('",ms_node_pairs[j][3],"','",ms_node_pairs[j][4],"' with loss difference: ",loss_diff[i][j][k],' > Threshold of ',threshold)
                    anomoly_df = anomoly_df.append({
                                    #'id': str(j),
                                'source': ms_node_pairs[j][3],
                                'destination': ms_node_pairs[j][4],
                                'threshold': threshold,
                                'prediction_error': loss_diff[i][j][k]},ignore_index=True)   
            if count > 0:
                anomoly_df.to_csv('output/V'+str(model_num)+'_timeslot_'+str(i+1)+'_AnomolyDetectionResults.csv',sep=',', encoding='utf-8')
        
        
        
        evl = evaluation(x_test[0:len_test, step_idx + n_his, :, :], y_test, x_stats)
        #print('evl.shape=',evl.shape) #ex: 9
        #print('evl=',evl)
        print('Final results, for each tuple records test comparison or evaluation results given below')
        
        if inf_mode == 'sep':
            te = evl
            #print(f'Time Step {tmp_idx}: MAPE {te[0]:7.3}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
            print(f'Time Step {tmp_idx}: MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
            
        if inf_mode == 'merge':
            for ix in tmp_idx:
                te = evl[ix - 2:ix + 1]
                #print(f'Time Step {ix + 1}: MAPE {te[0]:7.3}; MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
                print(f'Time Step {ix + 1}: MAE  {te[1]:4.3f}; RMSE {te[2]:6.3f}.')
                
        print(f'Model Test Time {time.time() - start_time:.3f}s')
    print('Testing model finished!')

