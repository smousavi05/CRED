#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:44:14 2018

@author: mostafamousavi
"""
from __future__ import print_function
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import csv
import h5py
import time
import datetime
from keras.models import load_model
np.seterr(divide='ignore', invalid='ignore')
from cred_utils import DataGenerator, model_cred, lr_schedule, generate_arrays_from_file, detector, output_writter_test, plotter,  DataGenerator_test
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Inputs for CRED')    
parser.add_argument("--mode", dest='mode', default='test', help="train, test")
parser.add_argument("--data_dir", dest='data_dir', default="../../STEAD/dataset6/waveforms_12_20_19.hdf5", type=str, help="Input file directory") #"../STEAD/dataset/waveforms.hdf5"  "./input/20101001WHAR.hdf5"
parser.add_argument("--data_list", dest='data_list', default="../../STEAD/dataset6/metadata_12_20_19.csv", type=str, help="Input csv file")
parser.add_argument("--input_model", dest='input_model', default= "./cred_original_outputs/models/cred_original_006.h5", type=str, help="The pre-trained model used for the prediction")
parser.add_argument("--input_testset", dest='input_testset', default= "./cred_original_outputs/test.npy", type=str, help="List of trace names for test.")
parser.add_argument("--output_dir", dest='output_dir', default='cred_original_test', type=str, help="Output directory")
parser.add_argument("--batch_size", dest='batch_size', default= 500, type=int, help="batch size")  
parser.add_argument("--epochs", dest='epochs', default= 200, type=int, help="number of epochs (default: 100)")
parser.add_argument('--gpuid', dest='gpuid', type=int, default=None, help='specifyin the GPU')
parser.add_argument('--gpu_limit', dest='gpu_limit', type=float, default=0.8, help='limiting the GPU memory')
parser.add_argument("--input_dimention", dest='input_dimention', default=(151, 41), type=int, help="a tuple including the time series lenght and number of channels.")  
parser.add_argument("--shuffle", dest='shuffle', default= True, type=bool, help="shuffling the list during the preprocessing and training")
parser.add_argument("--label_type",dest='label_type',  default='triangle', type=str, help="label type for picks: 'gaussian', 'triangle', 'box' ") 
parser.add_argument("--normalization_mode", dest='normalization_mode', default='max', type=str, help="normalization mode for preprocessing: 'std' or 'max' ") 
parser.add_argument("--augmentation", dest='augmentation', default= True, type=bool, help="if True, half of each batch will be augmented")  
parser.add_argument("--add_event_r", dest='add_event_r', default= 0.5, type=float, help=" chance for randomly adding a second event into the waveform") 
parser.add_argument("--shift_event_r", dest='shift_event_r', default= 0.9, type=float, help=" shift the event") 
parser.add_argument("--add_noise_r", dest='add_noise_r', default= 0.4, type=float, help=" chance for randomly adding Gaussian noise into the waveform")  
parser.add_argument("--scale_amplitude_r", dest='scale_amplitude_r', default= None, type=float, help=" chance for randomly amplifying the waveform amplitude ") 
parser.add_argument("--pre_emphasis", dest='pre_emphasis', default= False, type= bool, help=" if raw waveform needs to be pre emphesis ")
parser.add_argument("--train_valid_test_split", dest='train_valid_test_split', default=[0.85, 0.05, 0.10], type= float, help=" precentage for spliting the data into training, validation, and test sets")  
parser.add_argument("--patience", dest='patience', default= 5, type= int, help=" patience for early stop monitoring ") 
parser.add_argument("--detection_threshold",dest='detection_threshold',  default=0.05, type=float, help="Probability threshold for P pick")
parser.add_argument("--report", dest='report', default=True, type=bool, help="summary of the training settings and results.")
parser.add_argument("--plot_num", dest='plot_num', default= 100, type=int, help="number of plots for the test or predition results")
args = parser.parse_args()



print('Mode is: {}'.format(args.mode))    
if args.gpuid:           
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.gpuid)
    tf.Session(config=tf.ConfigProto(log_device_placement=True))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = float(args.gpu_limit) 
    K.tensorflow_backend.set_session(tf.Session(config=config))

save_dir = os.path.join(os.getcwd(), str(args.output_dir)+'_outputs')
         
if args.mode == 'train':
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir) 
    save_models = os.path.join(save_dir, 'models')
    os.makedirs(save_models)
    model_name = str(args.output_dir)+'_{epoch:03d}.h5' 
    filepath = os.path.join(save_models, model_name)

######################### BUILDING THE MODEL ##################################
    model = model_cred((151, 41, 3), filters = [8, 16, 32, 64, 128, 256])
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['binary_accuracy'])
    model.summary()
    
###############################################################################
#################################################################### Traininng
    df = pd.read_csv(args.data_list)
    ev_list = df.trace_name.tolist()
  #  ev_list = ev_list[:10000]  #--------------------     
    np.random.shuffle(ev_list)     
    training = ev_list[:int(args.train_valid_test_split[0]*len(ev_list))]
    validation =  ev_list[int(args.train_valid_test_split[0]*len(ev_list)):
                                int(args.train_valid_test_split[0]*len(ev_list) + args.train_valid_test_split[1]*len(ev_list))]
    test =  ev_list[ int(args.train_valid_test_split[0]*len(ev_list) + args.train_valid_test_split[1]*len(ev_list)):]
    np.save(save_dir+'/test', test) 
           
    params = {'file_name': args.data_dir,
              'dim': args.input_dimention,
              'batch_size': args.batch_size,
              'n_channels': 3,
              'target_length': 38,
              'shuffle': True,
              'norm_mode': args.normalization_mode,
              'augmentation': args.augmentation,
              'add_event_r': args.add_event_r,  
              'shift_event_r': args.shift_event_r,  
              'add_noise_r': args.add_noise_r, 
              'scale_amplitude_r': args.scale_amplitude_r,
              'pre_emphasis': args.pre_emphasis
              }  

            
    training_generator = DataGenerator(training, **params)
    validation_generator = DataGenerator(validation, **params)
    
    early_stopping_monitor = EarlyStopping(patience= args.patience)
    
    checkpoint = ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 mode = 'auto',
                                 verbose=1,
                                 save_best_only=True)
    
    lr_scheduler = LearningRateScheduler(lr_schedule)
    
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                   cooldown=0,
                                   patience=args.patience-2,
                                   min_lr=0.5e-6)
    
    callbacks = [checkpoint, lr_reducer, lr_scheduler, early_stopping_monitor]  
    start_time = time.time()          
    history = model.fit_generator(generator = training_generator,
                                  validation_data= validation_generator,
                                  max_queue_size=5,
                                  shuffle=False, 
                                  epochs= args.epochs,
                                  callbacks = callbacks)
    end_time = time.time()          
    np.save(save_dir+'/history',history)     
    
    model.save(save_dir+'/final_model.h5')
    model.to_json()   
    model.save_weights(save_dir+'/model_weights.h5')
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'], '--')
    ax.legend(['loss', 'val_loss'], loc='upper right')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_loss.png'))) 
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(history.history['binary_accuracy'])
    ax.plot(history.history['val_binary_accuracy'], '--')
    ax.legend(['accuracy', 'val_accuracy'], loc='upper right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    fig.savefig(os.path.join(save_dir,str('X_learning_curve_accuracy.png')))   
    
    
    
    
    
###############################################################################
####################################################################### Testing 
elif args.mode == 'test':
    
    save_figs = os.path.join(save_dir, 'figures')
    if os.path.isdir(save_figs):
        shutil.rmtree(save_figs) 
    os.makedirs(save_figs) 
    
    csvTst = open(os.path.join(save_dir,'X_test_results.csv'), 'w')          
    test_writer = csv.writer(csvTst, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    test_writer.writerow(['network_code', 
                          'ID', 
                          'earthquake_distance_km', 
                          'snr_db', 
                          'trace_name', 
                          'trace_category', 
                          'trace_start_time', 
                          'source_magnitude', 
                          'p_arrival_sample',
                          'p_status', 
                          'p_weight',
                          's_arrival_sample', 
                          's_status', 
                          's_weight', 
                          'receiver_type',
                          
                          'number_of_detections',
                          'detection_probability'
                          ])  
    csvTst.flush()        
        
    plt_n = 0
    test = np.load(args.input_testset)
    
    print('Loading the model ...')        
    model = load_model(args.input_model)
    
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr_schedule(0)),
                  metrics=['binary_accuracy'])
    print('Loading is complete!')  
    print('Testing ...')    
    print('Writting results into: " ' + str(args.output_dir)+'_outputs'+' "')
    start_time = time.time()          
    plt_n = 0
    list_generator = generate_arrays_from_file(test, args.batch_size)  
    pbar_test = tqdm(total=int(np.ceil(len(test)/args.batch_size))) 
    for _ in range(int(np.ceil(len(test) / args.batch_size))):
        pbar_test.update()
        new_list = next(list_generator)  
              
        params_test = {'file_name': args.data_dir,
                      'batch_size': args.batch_size} 
        
        test_generator = DataGenerator_test(new_list, **params_test)    
        predD = model.predict_generator(generator=test_generator)
             
        
        test_set={}
        fl = h5py.File(args.data_dir, 'r')
        for ID in new_list:
            if ID.split('_')[-1] == 'EV':
                dataset = fl.get('earthquake/local/'+str(ID))
            elif ID.split('_')[-1] == 'NO':
                dataset = fl.get('non_earthquake/noise/'+str(ID))
            test_set.update( {str(ID) : dataset})                 
        
        if len(predD) > 0:
            for ts in range(predD.shape[0]):
                evi =  new_list[ts] 
                dataset = test_set[evi]  
                try:
                    spt = int(dataset.attrs['p_arrival_sample']);
                except Exception:     
                    spt = None
                try:
                    sst = int(dataset.attrs['s_arrival_sample']);
                except Exception:     
                    sst = None                 
                matches = detector(args, predD[ts]) 
                output_writter_test(args, 
                                   dataset, 
                                   evi, 
                                   test_writer,
                                   csvTst,
                                   matches)
                            
                if plt_n < args.plot_num:  
                    plotter(ts, 
                           dataset,
                           evi,
                           args, 
                           save_figs, 
                           predD[ts], 
                           matches)
                    plt_n += 1;   
                
    end_time = time.time()          
    


if args.report:
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))
    delta = end_time - start_time;
    hour = int(delta / 3600)
    delta -= hour * 3600
    minute = int(delta / 60)
    delta -= minute * 60
    seconds = delta  
    
    with open(os.path.join(save_dir,'X_report.txt'), 'a') as the_file:            
        the_file.write('date: '+str(datetime.datetime.now())+'\n')         
        the_file.write('mode: '+str(args.mode)+'\n')   
        the_file.write('data_dir: '+str(args.data_dir)+'\n')            
        the_file.write('data_list: '+str(args.data_list)+'\n')
        the_file.write('input_model: '+str(args.input_model)+'\n')  
        the_file.write('output_dir: '+str(args.output_dir+'_outputs')+'\n')                     
        the_file.write('batch_size: '+str(args.batch_size)+'\n')

        if args.mode == 'train':                      
            the_file.write('@@@@@@@@@@@@@@@@@ Training Parameters @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'+'\n')  
            the_file.write('epochs: '+str(args.epochs)+'\n')
            the_file.write(f'finished the training in: {hour} hours and {minute} minutes and {round(seconds,2)} seconds\n') 
                   
            the_file.write('total number of events: '+str(len(ev_list))+'\n')
            the_file.write('train_valid_test_split: '+str(args.train_valid_test_split)+'\n')           
            the_file.write('total number of training: '+str(len(training))+'\n')
            the_file.write('total number of validation: '+str(len(validation))+'\n')
            the_file.write('total number of test: '+str(len(test))+'\n')  
            the_file.write('patience: '+str(args.patience)+'\n') 
            the_file.write('stoped after epoche: '+str(len(history.history['loss']))+'\n')
            the_file.write('last loss: '+str(history.history['loss'][-1])+'\n')
            the_file.write('last accuracy: '+str(history.history['binary_accuracy'][-1])+'\n')
            
            the_file.write('@@@@@@@@@@@@@@@@@ Model Parameters @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@'+'\n')            
            the_file.write(str('Total params: {:,}'.format(trainable_count + non_trainable_count))+'\n')    
            the_file.write(str('Trainable params: {:,}'.format(trainable_count))+'\n')    
            the_file.write(str('Non-trainable params: {:,}'.format(non_trainable_count))+'\n')                
            the_file.write('normalization_mode: '+str(args.normalization_mode)+'\n')                
            
        elif args.mode == 'test':     
            the_file.write(f'finished testing in: {hour} hours and {minute} minutes and {round(seconds,2)} seconds\n') 
            the_file.write('detection_threshold: '+str(args.detection_threshold)+'\n')            
            the_file.write('plot_num: '+str(args.plot_num)+'\n')   
        else:
            print('Please define the mode!')
             

   
    
    
    