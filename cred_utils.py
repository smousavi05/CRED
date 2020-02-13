#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 23:15:52 2020

@author: mostafamousavi
"""
from __future__ import print_function
import keras
from keras.layers import add, Reshape, Dense,Input, TimeDistributed, Dropout, Activation, LSTM, Conv2D, Bidirectional, BatchNormalization 
from keras.regularizers import l1
from keras import backend as K
from keras.models import Model
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import os
np.seterr(divide='ignore', invalid='ignore')
import h5py
from obspy.signal.trigger import trigger_onset
np.warnings.filterwarnings('ignore')

###############################################################################
###################################################################  Generator

class DataGenerator(keras.utils.Sequence):
    
    """ Keras generator with preprocessing 
    Args:
        list_IDsx: list of waveform names, str
        file_name: name of hdf file containing waveforms data, str
        dim: waveform lenght, int       
        batch_size: batch size, int
        n_channels: number of channels, int
        phase_window: number of samples (window) around each phase, int
        shuffle: shuffeling the list, boolean
        norm_mode: normalization type, str
        augmentation: if augmentation is True, half of each batch will be augmented version of the other half, boolean
        add_event_r: chance for randomly adding a second event into the waveform, float
        add_noise_r: chance for randomly adding Gaussian noise into the waveform, float
        drop_channe_r: chance for randomly dropping some of the channels, float
        scale_amplitude_r: chance for randomly amplifying the waveform amplitude, float
        pre_emphasis: if raw waveform needs to be pre emphesis,  boolean

    Returns:
        Batches of two dictionaries:
        {'input': X}: pre-processed waveform as input
        {'detector': y1, 'picker_P': y2, 'picker_S': y3}: outputs including three separate numpy arrays as labels for detection,
            P, and S respectively.
    """   
    def __init__(self, 
                 list_IDs, 
                 file_name, 
                 dim=(151, 41), 
                 batch_size=32, 
                 n_channels=3, 
                 target_length =38, 
                 shuffle=True, 
                 norm_mode = 'max',
                 augmentation = True, 
                 add_event_r = None,
                 shift_event_r = None,
                 add_noise_r = None, 
                 scale_amplitude_r = None, 
                 pre_emphasis = False):
        
        'Initialization'
        self.list_IDs = list_IDs
        self.file_name = file_name           
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.target_length = target_length
        self.shuffle = shuffle
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.augmentation = augmentation   
        self.add_event_r = add_event_r 
        self.shift_event_r = shift_event_r
        self.add_noise_r = add_noise_r
        self.scale_amplitude_r = scale_amplitude_r
        self.pre_emphasis = pre_emphasis

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)      
        
    def _normalize(self, data, mode = 'max'):
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            data /= max_data    
        elif mode == 'std':        
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data
        
    def _scale_amplitude(self, data, rate):
        tmp = np.random.uniform(0, 1)
        if tmp < rate:
            data *= np.random.uniform(1, 3)
        elif tmp < 2*rate:
            data /= np.random.uniform(1, 3)
        return data

    def _add_noise(self, data, snr, rate):
        data_noisy = np.empty((data.shape))
        if np.random.uniform(0, 1) < rate and all(snr >= 5.0): 
            data_noisy = np.empty((data.shape))
            noise = np.random.normal(0,1,data.shape[0])
            data_noisy[:, 0] = data[:,0] + 0.5*(noise*(10**(snr[0]/10)))* np.random.random()
            data_noisy[:, 1] = data[:,1] + 0.5*(noise*(10**(snr[1]/10)))* np.random.random()
            data_noisy[:, 2] = data[:,2] + 0.5*(noise*(10**(snr[2]/10)))* np.random.random()    
        else:
            data_noisy = data
        return data_noisy  

    def _add_event(self, data, addp, adds, coda_end, snr, rate): 
        added = np.copy(data)
        additions = None
        spt_secondEV = None
        sst_secondEV = None
        if addp and adds:
            s_p = adds - addp
            if np.random.uniform(0, 1) < rate and all(snr >= 5.0) and (data.shape[0]-s_p-21-coda_end) > 20:     
                secondEV_strt = np.random.randint(coda_end, data.shape[0]-s_p-21)
                space = data.shape[0]-secondEV_strt  
                added[secondEV_strt:secondEV_strt+space, 0] += data[addp:addp+space, 0]*np.random.uniform(0, 1)
                added[secondEV_strt:secondEV_strt+space, 1] += data[addp:addp+space, 1]*np.random.uniform(0, 1) 
                added[secondEV_strt:secondEV_strt+space, 2] += data[addp:addp+space, 2]*np.random.uniform(0, 1)          
                spt_secondEV = secondEV_strt   
                if  spt_secondEV + s_p + 21 <= data.shape[0]:
                    sst_secondEV = spt_secondEV + s_p
                if spt_secondEV and sst_secondEV:                                                                     
                    additions = [spt_secondEV, sst_secondEV] 
                    data = added
                 
        return data, additions    
    
    def _shift_event(self, data, addp, adds, coda_end, snr, rate): 
        org_len = len(data) 
        data2 = None;
        addp2 = None;
        adds2 = None;
        coda_end2 = None;          
        if np.random.uniform(0, 1) < rate and all(snr >= 5.0):
            
            space = int(org_len - coda_end)
            preNoise = int(addp)-100 
            
            noise0 = data[:preNoise, :];
            noise1 = noise0; 
            if preNoise > 0:
                repN = int(np.floor(space/preNoise))-1            
            
                if repN >= 5:
                    for _ in range(np.random.randint(1, repN)):        
                        noise1 = np.concatenate([noise1, noise0], axis=0)
                else: 
                    for _ in range(repN):        
                        noise1 = np.concatenate([noise1, noise0], axis=0)                
                    
                data2 = np.concatenate([noise1, data], axis=0)
                data2 = data2[:org_len, :];
                if addp+len(noise1) >= 0 and addp+len(noise1) < org_len:
                    addp2 = addp+len(noise1);
                else:
                    addp2 = None;
                    
                if adds+len(noise1) >= 0 and adds+len(noise1) < org_len:               
                    adds2 = adds+len(noise1);
                else:
                    adds2 = None;
                    
                if coda_end+len(noise1) < org_len:                              
                    coda_end2 = coda_end+len(noise1) 
                else:
                    coda_end2 = org_len 
                
                if addp2 and adds2:
                    data = data2;
                    addp = addp2;
                    adds = adds2;
                    coda_end= coda_end2; 
                                    
        return data, addp, adds, coda_end     
    

    def _pre_emphasis(self, data, pre_emphasis = 0.97):
        for ch in range(self.n_channels): 
            bpf = data[:, ch]  
            data[:, ch] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data

  
    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        y = np.zeros((self.batch_size, 38, 1))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            try:
                if ID.split('_')[-1] == 'EV':
                    dataset = fl.get('earthquake/local/'+str(ID))
                    data = np.array(dataset)                   
                    snr = dataset.attrs['snr_db'];
                    coda_end = int(dataset.attrs['coda_end_sample']);
                    spt = int(dataset.attrs['p_arrival_sample']);
                    sst = int(dataset.attrs['s_arrival_sample']);
                       
                elif ID.split('_')[-1] == 'NO':
                    dataset = fl.get('non_earthquake/noise/'+str(ID))
                    data = np.array(dataset)
                      
                ## augmentation 
                if self.augmentation == True:                 
                    if i <= self.batch_size//2:   
                        if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local' and all(snr):
                            data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                                       
                        if self.norm_mode:                    
                            data = self._normalize(data, self.norm_mode)  
                    else:                  
                        if dataset.attrs['trace_category'] == 'earthquake_local':                   
                            if self.shift_event_r and all(snr):
                                data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r); 
                                
                            if self.add_event_r and all(snr):
                                data, additions = self._add_event(data, spt, sst, coda_end, snr, self.add_event_r); 
                                    
                            if self.add_noise_r and all(snr):
                                data = self._add_noise(data, snr, self.add_noise_r);
                                        
                            if self.scale_amplitude_r:
                                data = self._scale_amplitude(data, self.scale_amplitude_r); 
                                        
                            if self.pre_emphasis:  
                                data = self._pre_emphasis(data) 
                                        
                            if self.norm_mode:    
                                data = self._normalize(data, self.norm_mode)                            
                                        
                        elif dataset.attrs['trace_category'] == 'noise':
                            if self.norm_mode:                    
                                data = self._normalize(data, self.norm_mode) 
    
                elif self.augmentation == False:  
                    if self.shift_event_r and dataset.attrs['trace_category'] == 'earthquake_local' and all(snr):
                        data, spt, sst, coda_end = self._shift_event(data, spt, sst, coda_end, snr, self.shift_event_r/2);                     
                    if self.norm_mode:                    
                        data = self._normalize(data, self.norm_mode) 
                                    
                if not np.any(np.isnan(data).any()):
                    for ch in range(self.n_channels): 
                        bpf = data[:, ch]                        
                        f, t, Pxx = signal.stft(bpf, fs = 100, nperseg=80)
                        Pxx = np.abs(Pxx)
                        
                        X[i, :, :, ch] = Pxx.T 
                             
                    # making labels for detection
                    if ID.split('_')[-1] == 'EV':
                        sptS = int(spt*self.target_length/6000);
                        sstS = int(sst*self.target_length/6000);                
                        delta = sstS - sptS                
                        y[i, sptS:int(sstS+(1.2*delta)), 0] = 1
                        
            except Exception:
                pass
        assert not np.any(np.isnan(X))                    
        return X.astype('float32'), y.astype('float32')
    



def generate_arrays_from_file(file_list, step):
    n_loops = int(np.ceil(len(file_list) / step))
    b = 0
    while True:
        for i in range(n_loops):
            e = i*step + step 
            if e > len(file_list):
                e = len(file_list)
            chunck = file_list[b:e]
            b=e
            yield chunck





class DataGenerator_test(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, file_name, batch_size=32, dim=(151, 41), n_channels=3, norm_mode = 'max'):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.on_epoch_end()
        self.norm_mode = norm_mode
        self.file_name = file_name        

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        
    def _normalize(self, data, mode = 'max'):
        data -= np.mean(data, axis=0, keepdims=True)
        if mode == 'max':
            max_data = np.max(data, axis=0, keepdims=True)
            assert(max_data.shape[-1] == data.shape[-1])
            data /= max_data    
        elif mode == 'std':        
            std_data = np.std(data, axis=0, keepdims=True)
            assert(std_data.shape[-1] == data.shape[-1])
            std_data[std_data == 0] = 1
            data /= std_data
        return data

    def __data_generation(self, list_IDs_temp):
        'readint the waveforms' 
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        fl = h5py.File(self.file_name, 'r')

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            if ID.split('_')[-1] == 'EV':
                dataset = fl.get('earthquake/local/'+str(ID))
                data = np.array(dataset)                   
                   
            elif ID.split('_')[-1] == 'NO':
                dataset = fl.get('non_earthquake/noise/'+str(ID))
                data = np.array(dataset)            

            data = self._normalize(data, self.norm_mode) 
                           
            for ch in range(self.n_channels): 
                bpf = data[:, ch]                        
                f, t, Pxx = signal.stft(bpf, fs = 100, nperseg=80)
                Pxx = np.abs(Pxx)
                X[i, :, :, ch] = Pxx.T 
                        
        return X.astype('float32')    
    
    
    


def detector(args, yh1):

    """
    return two dictionaries and one numpy array:
        
        matches --> {detection statr-time:[ detection end-time,
                                           detection probability,
                                           
                                           ]}
                
    """               
             
    detection = trigger_onset(yh1, args.detection_threshold, args.detection_threshold/2)

    EVENTS = {}
    matches = {}
                       
    if len(detection) > 0:        
        for ev in range(len(detection)):                                 
            D_prob = np.mean(yh1[detection[ev][0]:detection[ev][1]])
            D_prob = np.round(D_prob, 3)                  
            EVENTS.update({ detection[ev][0] : [D_prob, detection[ev][1]]})            

    for ev in EVENTS:
        bg = ev
        ed = EVENTS[ev][1]
        
        if int(ed-bg) >= 1:               
            matches.update({ bg:[ed, EVENTS[ev][0]
                                                ] })                                               
    return matches


 
    
def output_writter_test(args, 
                        dataset, 
                        evi, 
                        output_writer, 
                        csvfile, 
                        matches 
                        ):
    
    numberOFdetections = len(matches)
    
    if numberOFdetections != 0: 
        D_prob =  matches[list(matches)[0]][1]
    else: 
        D_prob = None

    
    if evi.split('_')[-1] == 'EV':                                     
        network_code = dataset.attrs['network_code']
        source_id = dataset.attrs['source_id']
        source_distance_km = dataset.attrs['source_distance_km']  
        snr_db = np.mean(dataset.attrs['snr_db'])
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = dataset.attrs['trace_start_time'] 
        source_magnitude = dataset.attrs['source_magnitude'] 
        p_arrival_sample = dataset.attrs['p_arrival_sample'] 
        p_status = dataset.attrs['p_status'] 
        p_weight = dataset.attrs['p_weight'] 
        s_arrival_sample = dataset.attrs['s_arrival_sample'] 
        s_status = dataset.attrs['s_status'] 
        s_weight = dataset.attrs['s_weight'] 
        receiver_type = dataset.attrs['receiver_type']  
                   
    elif evi.split('_')[-1] == 'NO':               
        network_code = dataset.attrs['network_code']
        source_id = None
        source_distance_km = None 
        snr_db = None
        trace_name = dataset.attrs['trace_name'] 
        trace_category = dataset.attrs['trace_category']            
        trace_start_time = None
        source_magnitude = None
        p_arrival_sample = None
        p_status = None
        p_weight = None
        s_arrival_sample = None
        s_status = None
        s_weight = None
        receiver_type = dataset.attrs['receiver_type'] 

    output_writer.writerow([network_code, 
                            source_id, 
                            source_distance_km, 
                            snr_db, 
                            trace_name, 
                            trace_category, 
                            trace_start_time, 
                            source_magnitude,
                            p_arrival_sample, 
                            p_status, 
                            p_weight, 
                            s_arrival_sample, 
                            s_status,
                            s_weight,
                            receiver_type,                
                            numberOFdetections,
                            D_prob
                            
                            ]) 
    
    csvfile.flush()             
    





def plotter(ts, 
            dataset, 
            evi,
            args,
            save_figs,
            yh1,
            matches
            ):

    try:
        spt = int(dataset.attrs['p_arrival_sample']);
    except Exception:     
        spt = None
                    
    try:
        sst = int(dataset.attrs['s_arrival_sample']);
    except Exception:     
        sst = None
    
    data = np.array(dataset)
    
    fig = plt.figure()
    ax = fig.add_subplot(411)         
    plt.plot(data[:, 0], 'k')
    plt.rcParams["figure.figsize"] = (8,5)
    legend_properties = {'weight':'bold'}  
    plt.title(str(evi))
    plt.tight_layout()
    ymin, ymax = ax.get_ylim() 
    pl = None
    sl = None       

    
    if dataset.attrs['trace_category'] == 'earthquake_local':
        if dataset.attrs['p_status'] == 'manual':
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_P_Arrival')
        else:
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Auto_P_Arrival')
            
        if dataset.attrs['s_status'] == 'manual':
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_S_Arrival')
        else:
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Auto_S_Arrival')
        if pl or sl:    
            plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)     
                            
    ax = fig.add_subplot(412)   
    plt.plot(data[:, 1] , 'k')
    plt.tight_layout()                
    if dataset.attrs['trace_category'] == 'earthquake_local':
        if dataset.attrs['p_status'] == 'manual':
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_P_Arrival')
        else:
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Auto_P_Arrival')
            
        if dataset.attrs['s_status'] == 'manual':
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_S_Arrival')
        else:
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Auto_S_Arrival')
        if pl or sl:    
            plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)    

    ax = fig.add_subplot(413) 
    plt.plot(data[:, 2], 'k')   
    plt.tight_layout()                
    if dataset.attrs['trace_category'] == 'earthquake_local':
        if dataset.attrs['p_status'] == 'manual':
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Manual_P_Arrival')
        else:
            pl = plt.vlines(int(spt), ymin, ymax, color='b', linewidth=2, label='Auto_P_Arrival')
            
        if dataset.attrs['s_status'] == 'manual':
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Manual_S_Arrival')
        else:
            sl = plt.vlines(int(sst), ymin, ymax, color='r', linewidth=2, label='Auto_S_Arrival')
        if pl or sl:    
            plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties)                 
    ax = fig.add_subplot(414)
    plt.plot(yh1, 'g--', alpha = 0.5, linewidth=1.5, label='Detection')
    plt.tight_layout()       
    plt.ylim((-0.1, 1.1))
    plt.legend(loc = 'upper right', borderaxespad=0., prop=legend_properties) 
                        
    fig.savefig(os.path.join(save_figs, str(evi.split('/')[-1])+'.png')) 


  
 
    

############################################################# model

def lr_schedule(epoch):
    """
    Learning rate is scheduled to be reduced after 40, 60, 80, 90 epochs.
    """
    lr = 1e-3
    if epoch > 60:
        lr *= 0.5e-3
    elif epoch > 40:
        lr *= 1e-3
    elif epoch > 20:
        lr *= 1e-2
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def block_CNN(filters, ker, inpC): 
    """
    Returns CNN residual blocks
    """
    layer_1 = BatchNormalization()(inpC) 
    act_1 = Activation('relu')(layer_1) 

    conv_1 = Conv2D(filters, (ker-2, ker-2), padding = 'same')(act_1) 
    
    layer_2 = BatchNormalization()(conv_1) 
    act_2 = Activation('relu')(layer_2) 
  
    conv_2 = Conv2D(filters, (ker-2, ker-2), padding = 'same')(act_2) 
    return(conv_2) 



def block_BiLSTM(inpR, filters, rnn_depth):
    """
    Returns LSTM residual blocks
    """
    x = inpR
    for i in range(rnn_depth):
        x_rnn = Bidirectional(LSTM(filters, return_sequences=True))(x)
        x_rnn = Dropout(0.7)(x_rnn)
        if i > 0 :
           x = add([x, x_rnn])
        else:
           x = x_rnn      
    return x
     

def model_cred(shape, filters):
    
    inp = Input(shape=shape, name='input')

    conv2D_2 = Conv2D(filters[0], (9,9), strides = (2,2), padding = 'same', activation = 'relu')(inp) 
    res_conv_2 = keras.layers.add([block_CNN(filters[0], 9, conv2D_2), conv2D_2]) 

    conv2D_3 = Conv2D(filters[1], (5,5), strides = (2,2), padding = 'same', activation = 'relu')(res_conv_2) 
    res_conv_3 = keras.layers.add([block_CNN(filters[1], 5, conv2D_3),conv2D_3]) 
    
    shape = K.int_shape(res_conv_3)   
    reshaped = Reshape((shape[1], shape[2]*shape[3]))(res_conv_3)
    
    res_BIlstm = block_BiLSTM(reshaped, filters = filters[3], rnn_depth = 2)
 
    UNIlstm = LSTM(filters[3], return_sequences=True)(res_BIlstm)
    UNIlstm = Dropout(0.8)(UNIlstm)  
    UNIlstm = BatchNormalization()(UNIlstm)
   
    dense_2 = TimeDistributed(Dense(filters[3], kernel_regularizer=l1(0.01), activation='relu'))(UNIlstm)
    dense_2 = BatchNormalization()(dense_2)
    dense_2 = Dropout(0.8)(dense_2)
    
    dense_3 = TimeDistributed(Dense(1, kernel_regularizer=l1(0.01), activation='sigmoid'))(dense_2)

    out_model = Model(inputs=inp, outputs=dense_3)
    return out_model     