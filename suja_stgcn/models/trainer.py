from data_loader.data_utils import gen_batch
from data_loader.data_utils import writeToCSV
from models.tester import model_inference
from models.base_model import build_model, model_save
from os.path import join as pjoin
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import time
import os
import sys

def plot_eval_metrics(training_loss, validating_loss, test_num):
        # define and create a MAE metric directory
        plot_dir = 'mae_loss_plots/'
        loss_file = 'V'+str(test_num)+'_mae_training_loss.png'
        file_name = os.path.join(plot_dir, loss_file)
        #os.makedirs(loss_dir)

        # define a metric file
        
        if os.path.exists(file_name):
            os.remove(file_name)

        plt.figure()
        plt.plot(training_loss, label='training_loss')
        #plt.plot(validating_loss, label='validating loss')
        plt.title('STGCN - Chart for training loss - Test No - '+str(test_num))
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(file_name)
        plt.show()

def plot_mae_validation_loss(validating_loss, test_num):
        # define and create a MAE metric directory
        plot_dir = 'mae_loss_plots/'
        loss_file = 'V'+str(test_num)+'_mae_validation_loss.png'
        file_name = os.path.join(plot_dir, loss_file)
        #os.makedirs(loss_dir)

        # define a metric file
        
        if os.path.exists(file_name):
            os.remove(file_name)

        plt.figure()
        #plt.plot(training_loss, label='training_loss')
        plt.plot(validating_loss, label='validating loss')
        plt.title('STGCN - Chart for validation loss - Test No - '+str(test_num))
        plt.xlabel('epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(file_name)
        plt.show()

def model_train(inputs, blocks, args, test_number,sum_path='./output/tensorboard'):
    '''
    Train the base model.
    :param inputs: instance of class Dataset, data source for training.
    :param blocks: list, channel configs of st_conv blocks.
    :param args: instance of class argparse, args for training.
    '''
    n, n_his, n_pred = args.n_route, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, inf_mode, opt = args.batch_size, args.epoch, args.inf_mode, args.opt
    
    # define arrays for training and validation loss
    epoch_train_loss = []
    epoch_validate_loss = []

    # Placeholder for model training
    x = tf.compat.v1.placeholder(tf.float32, [None, n_his + 1, n, 1], name='data_input')
    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')
    #print('tf placeholder x(data_input) shape - ',x.shape)
    #print('tf placeholder keep_prob shape - ',keep_prob.shape)
    
    # Define model loss
    #print('building model.....using x,n_his,Ks,Kt,blocks,keep_prob',x,n_his,Ks,Kt,blocks,keep_prob)
    train_loss, pred = build_model(x, n_his, Ks, Kt, blocks, keep_prob)
    #print('train_loss shape',train_loss.shape)
    #print('pred shape',pred.shape)
    tf.compat.v1.summary.scalar('train_loss', train_loss)
    copy_loss = tf.add_n(tf.compat.v1.get_collection('copy_loss'))
    tf.compat.v1.summary.scalar('copy_loss', copy_loss)
    #print(copy_loss.shape)
    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    #print('Total training records = ',len_train)
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    print('number of epoch stpes = ',epoch_step)
    # Learning rate decay with rate 0.7 every 5 epochs.
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=5 * epoch_step, decay_rate=0.7, staircase=True)
    print('args.lr = ',args.lr,'global_steps = ',global_steps,'decay_steps = ',5*epoch_step,'decay_rate=0.7')
    tf.compat.v1.summary.scalar('learning_rate', lr)
    print('learning rate = ',lr)
    step_op = tf.compat.v1.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        if opt == 'RMSProp':
            train_op = tf.compat.v1.train.RMSPropOptimizer(lr).minimize(train_loss)
        elif opt == 'ADAM':
            train_op = tf.compat.v1.train.AdamOptimizer(lr).minimize(train_loss)
        else:
            raise ValueError(f'ERROR: optimizer "{opt}" is not defined.')

    merged = tf.compat.v1.summary.merge_all()

    with tf.compat.v1.Session() as sess:
        writer = tf.compat.v1.summary.FileWriter(pjoin(sum_path, 'train'), sess.graph)
        print('tensorflow summary graph is written in file - ',pjoin(sum_path, 'train'))
        sess.run(tf.compat.v1.global_variables_initializer())

        if inf_mode == 'sep':
            # for inference mode 'sep', the type of step index is int.
            step_idx = n_pred - 1
            tmp_idx = [step_idx]
            min_val = min_va_val = np.array([4e1, 1e5, 1e5])
        elif inf_mode == 'merge':
            # for inference mode 'merge', the type of step index is np.ndarray.
            step_idx = tmp_idx = np.arange(3, n_pred + 1, 3) - 1
            min_val = min_va_val = np.array([4e1, 1e5, 1e5] * len(step_idx))
        else:
            raise ValueError(f'ERROR: test mode "{inf_mode}" is not defined.')
        #print('step_index=',step_idx)
        #print('min_val=',min_val)
            
        for i in range(epoch):
            start_time = time.time()
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                summary, _ = sess.run([merged, train_op], feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                writer.add_summary(summary, i * epoch_step + j)
                #print('running epochstep=',epoch_step,' j=',j)
                if j % 50 == 0 : # or j==epoch_step-1 (added to ensure last block is also covered
                    loss_value = \
                        sess.run([train_loss, copy_loss],
                                 feed_dict={x: x_batch[:, 0:n_his + 1, :, :], keep_prob: 1.0})
                    print(f'Epoch {i:2d}, Step {j:3d}: [{loss_value[0]:.3f}, {loss_value[1]:.3f}]')
                    epoch_train_loss.append(loss_value[0])
                    #print('train_loss = ',train_loss) #train_loss =  Tensor("L2Loss_2:0", shape=(), dtype=float32)
                    #print('copy_loss = ',copy_loss) #copy_loss =  Tensor("L2Loss_1:0", shape=(), dtype=float32)
                    #print('loss_value = ',loss_value) #loss_value =  [51.06814, 45.636517] #This is same as what's printed on Epoch step above.
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            start_time = time.time()
            print('After every Epoch run, validate the data with so-far trained epoch cycles, validation cycle starts')
            min_va_val, min_val, betterPerformance = \
                model_inference(sess, pred, inputs, batch_size, n_his, n_pred, step_idx, min_va_val, min_val)
            #print('After calling new model inference, value of min_va_val=',min_va_val,' and min_val = ',min_val, ' and betterPerformance=',betterPerformance)

            if inf_mode == 'sep':
                va, te = min_va_val, min_val
                print(f'ValidationEval Step {tmp_idx}: '
                   #   f'MAPE {va[0]:7.3f}, {te[0]:7.3f}; '
                      f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                      f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
                epoch_validate_loss.append(va[1])
                
            if inf_mode == 'merge':
                i = 0
                for ix in tmp_idx:
                    i = i + 1
                    va, te = min_va_val[ix - 2:ix + 1], min_val[ix - 2:ix + 1]
                    #print(va)
                    #print(te)
                    print(f'ValidationEval Step {ix + 1}: '
                      #    f'MAPE {va[0]:7.3f}, {te[0]:7.3f}; '
                          f'MAE  {va[1]:4.3f}, {te[1]:4.3f}; '
                          f'RMSE {va[2]:6.3f}, {te[2]:6.3f}.')
                    if i == 1:
                        epoch_validate_loss.append(va[1])
                    
            
            
            print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')

            if (i + 1) % args.save == 0:# and betterPerformance: #make sure only the betterPerformance model is saved as the latest check point
                print('save the model, for every args.save==',args.save)
                model_save(sess, global_steps, 'STGCN')
        writer.close()
        plot_eval_metrics(epoch_train_loss, epoch_validate_loss,test_number)
        plot_mae_validation_loss(epoch_validate_loss,test_number)
    print('Training model finished!')
