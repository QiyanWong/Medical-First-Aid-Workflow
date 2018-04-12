# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import medfilt
from sklearn.manifold import TSNE
import csv
import pdb
def open_data(direc,dataset = "ECG5000",ratio_train = 0.8):
  """Input:
  direc: location of the UCR archive
  ratio_train: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""

  datadir = direc + '/' + dataset + '/' + dataset
  data_train = np.loadtxt(datadir+'_TRAIN',delimiter=',')
  data_test_val = np.loadtxt(datadir+'_TEST',delimiter=',')[:-1]
  data = np.concatenate((data_train,data_test_val),axis=0)

  N,D = data.shape

  ind_cut = int(ratio_train*N)
  ind = np.random.permutation(N)
  return 
  # data[ind[:ind_cut],1:],data[ind[ind_cut:],1:],data[ind[:ind_cut],0],data[ind[ind_cut:],0]

def open_data_list(list,ratio_train = 0.8):
  """Input:
  direc: location of the UCR archive
  ratio_train: ratio to split training and testset
  dataset: name of the dataset in the UCR archive"""
  # datadir = direc + '/' + dataset + '/' + dataset
  name_tmp = list[0].split("/")
  name = name_tmp[len(name_tmp)-1]
  data_train = np.loadtxt(list[0]+'/'+name+"_TRAIN",delimiter=',')
  data_test_val = np.loadtxt(list[0]+'/'+name+"_TEST",delimiter=',')[:-1]
  data = np.concatenate((data_train,data_test_val),axis=0)
  # print(data_train.shape)
  # print (data_test_val.shape)
  # ind_cut = int(ratio_train*N)
  # ind = np.random.permutation(N)

  return data_train[0:,1:],data_test_val[0:,1:],data_train[0:,0],data_test_val[0:,0]

def generate_data(train,label):
  list_tmp = []
  list_2 = np.power(train,2)
  list_3 = np.power(train,3)
  list_0_5 = np.sqrt(np.absolute(train))
  list_sin = np.sin(train)
  list_cos = np.cos(train)
  list_medfilt = medfilt(train)
  train = np.concatenate((train,list_2,list_3,list_0_5,list_sin,list_cos,list_medfilt))
  label = np.concatenate((label,label,label,label,label,label,label))
  return train, label

def open_csv(direc,list, add_length=True):
  res = []
  max_col = 0
  for i in range(len(list)):
    tmp = [[float(x) for x in rec] for rec in csv.reader(open(direc+list[i]), delimiter=',')]
    for each_row in tmp:
      max_col = np.amax((max_col,len(each_row)))
  res = np.array([], dtype=np.float32).reshape(0,max_col)
  res = []
  for i in range(len(list)):
    tmp = [[float(x) for x in rec] for rec in csv.reader(open(direc+list[i]), delimiter=',')]
    for each_row in tmp:
      if (max_col>len(each_row)):
        while(max_col>len(each_row)):
          each_row.append(0)
    tmp = np.asarray(tmp)
    tmp_t = []
    for col in range(len(tmp[0])):
      tmp_t.append(tmp[0:len(tmp),col])
    # print len(tmp_t),len(tmp_t[0])
    res.append(tmp_t)
  res = np.asarray(res)
  length = len(res)/10*8
  train = res[:length]
  test = res[length:]

  # res = []
  # for i in range(len(list)):
  #   f = open(direc+list[i])
  #   csv_f = csv.reader(f)
  #   for row in csv_f:
  #     print row
  return train,test, res
    

"""Plot the data"""
def plot_data(X_train, y_train, plot_row = 5):
  counts = dict(Counter(y_train))
  num_classes = len(np.unique(y_train))
  f, axarr = plt.subplots(plot_row, num_classes)
  for c in np.unique(y_train):    #Loops over classes, plot as columns
    c = int(c)
    ind = np.where(y_train == c)
    ind_plot = np.random.choice(ind[0],size=plot_row)
    for n in xrange(plot_row):  #Loops over rows
      axarr[n,c].plot(X_train[ind_plot[n],:])
      # Only shops axes for bottom row and left column
      if n == 0: axarr[n,c].set_title('Class %.0f (%.0f)'%(c,counts[float(c)]))
      if not n == plot_row-1:
        plt.setp([axarr[n,c].get_xticklabels()], visible=False)
      if not c == 0:
        plt.setp([axarr[n,c].get_yticklabels()], visible=False)
  f.subplots_adjust(hspace=0)  #No horizontal space between subplots
  f.subplots_adjust(wspace=0)  #No vertical space between subplots
  plt.show()
  return

def plot_patient(z_run):
  from sklearn.decomposition import TruncatedSVD
  f1, ax1 = plt.subplots(1, 1)
  PCA_model = TruncatedSVD(n_components=3).fit(z_run)
  z_run_reduced = PCA_model.transform(z_run)
  ax1.scatter(z_run_reduced[:,0],z_run_reduced[:,1],marker='*',linewidths = 0)
  ax1.set_title('PCA')
  f1.savefig("test.png")

def plot_z_run(z_run,label,clf="",each_data_set=None,accuracy=None):

  from sklearn.decomposition import TruncatedSVD
  f1, ax1 = plt.subplots(2, 1)
  plt. subplots_adjust(hspace=0.5)
  if(each_data_set==None):
    name=""
  else: 
    name_tmp = each_data_set[0].split("/")
    name = name_tmp[len(name_tmp)-1]
  PCA_model = TruncatedSVD(n_components=3).fit(z_run)
  z_run_reduced = PCA_model.transform(z_run)
  ax1[0].scatter(z_run_reduced[:,0],z_run_reduced[:,1],c=label,marker='*',linewidths = 0)
  ax1[0].set_title('PCA on '+clf+" for "+ name+", accuracy: %.4f"%accuracy)
  tSNE_model = TSNE(verbose=2, perplexity=30,min_grad_norm=1E-12,n_iter=3000)
  z_run_tsne = tSNE_model.fit_transform(z_run)
  ax1[1].scatter(z_run_tsne[:,0],z_run_tsne[:,1],c=label,marker='*',linewidths = 0)
  ax1[1].set_title('tSNE using '+clf+" for "+ name)
  f1.savefig("Save/Figure/"+name+clf+".png")
  return

class Model():
  def __init__(self,config):
    """Hyperparameters"""
    num_layers = config['num_layers']
    hidden_size = config['hidden_size']
    max_grad_norm = config['max_grad_norm']
    batch_size = config['batch_size']
    sl = config['sl']
    crd = config['crd']
    num_l = config['num_l']
    learning_rate = config['learning_rate']
    self.sl = sl
    self.batch_size = batch_size
    
    # Nodes for the input variables
    self.x = tf.placeholder("float", shape=[batch_size, sl,16], name = 'Input_data')
    self.x_exp = self.x#tf.expand_dims(self.x,1)
    # print self.x[0],self.x_exp[0]

    self.keep_prob = tf.placeholder("float")
    tf.get_variable_scope().reuse == True
    with tf.variable_scope("Encoder") as scope:  
      #Th encoder cell, multi-layered with dropout
      cell_enc = tf.nn.rnn_cell.LSTMCell(hidden_size)
      cell_enc = tf.nn.rnn_cell.MultiRNNCell([cell_enc] * num_layers)
      cell_enc = tf.nn.rnn_cell.DropoutWrapper(cell_enc,output_keep_prob=self.keep_prob)

      #Initial state
      initial_state_enc = cell_enc.zero_state(batch_size, tf.float32)
      # print (len(tf.unpack(self.x_exp,axis=)))
      outputs_enc,_ = tf.nn.seq2seq.rnn_decoder(tf.unpack(self.x_exp,axis=1),initial_state_enc,cell_enc)
      cell_output = outputs_enc[-1]  #Only use the final hidden state #tensor in [batch_size,hidden_size]
    # pdb.set_trace()
    with tf.name_scope("Enc_2_lat") as scope:
      #layer for mean of z
      W_mu = tf.get_variable('W_mu', [hidden_size,num_l])
      b_mu = tf.get_variable('b_mu',[num_l])
      self.z_mu = tf.nn.xw_plus_b(cell_output,W_mu,b_mu,name='z_mu')  #mu, mean, of latent space
      
      #Train the point in latent space to have zero-mean and unit-variance on batch basis
      lat_mean,lat_var = tf.nn.moments(self.z_mu,axes=[1])
      self.loss_lat_batch = tf.reduce_mean(tf.square(lat_mean)+lat_var - tf.log(lat_var)-1)
      
    with tf.name_scope("Lat_2_dec") as scope:
      #layer to generate initial state
      W_state = tf.get_variable('W_state', [num_l,hidden_size])
      b_state = tf.get_variable('b_state',[hidden_size])
      z_state = tf.nn.xw_plus_b(self.z_mu,W_state,b_state,name='z_state')  #mu, mean, of latent space
      
    with tf.variable_scope("Decoder") as scope:
      # The decoder, also multi-layered
      cell_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)
      cell_dec = tf.nn.rnn_cell.MultiRNNCell([cell_dec] * num_layers)

      #Initial state
      initial_state_dec = tuple([(z_state,z_state)]*num_layers)
      dec_inputs = [tf.zeros([batch_size,16])]*sl
      outputs_dec,_ = tf.nn.seq2seq.rnn_decoder(dec_inputs,initial_state_dec,cell_dec)
    with tf.name_scope("Out_layer") as scope:
      params_o = 2*crd   #Number of coordinates + variances
      W_o = tf.get_variable('W_o',[hidden_size,params_o])
      b_o = tf.get_variable('b_o',[params_o])
      outputs = tf.concat(0,outputs_dec)                    #tensor in [sl*batch_size,hidden_size]
      h_out = tf.nn.xw_plus_b(outputs,W_o,b_o)
      h_mu,h_sigma_log = tf.unstack(tf.reshape(h_out,[sl,batch_size,params_o]),axis=2)
      h_sigma = tf.exp(h_sigma_log)
      dist = tf.contrib.distributions.Normal(h_mu,h_sigma)
      px = dist.pdf(tf.transpose(self.x))
      loss_seq = -tf.log(tf.maximum(px, 1e-20))             #add epsilon to prevent log(0)
      self.loss_seq = tf.reduce_mean(loss_seq)
    pdb.set_trace()
    with tf.name_scope("train") as scope:
      #Use learning rte decay
      global_step = tf.Variable(0,trainable=False)
      lr = tf.train.exponential_decay(learning_rate,global_step,1000,0.1,staircase=False)
      
      
      self.loss = self.loss_seq + self.loss_lat_batch
      
      #Route the gradients so that we can plot them on Tensorboard
      tvars = tf.trainable_variables()
      #We clip the gradients to prevent explosion
      grads = tf.gradients(self.loss, tvars)
      grads, _ = tf.clip_by_global_norm(grads,max_grad_norm)
      self.numel = tf.constant([[0]])

      #And apply the gradients
      optimizer = tf.train.AdamOptimizer(lr)
      gradients = zip(grads, tvars)
      self.train_step = optimizer.apply_gradients(gradients,global_step=global_step)
#      for gradient, variable in gradients:  #plot the gradient of each trainable variable
#        if isinstance(gradient, ops.IndexedSlices):
#          grad_values = gradient.values
#        else:
#          grad_values = gradient
#
#        self.numel +=tf.reduce_sum(tf.size(variable))
#        tf.summary.histogram(variable.name, variable)
#        tf.summary.histogram(variable.name + "/gradients", grad_values)
#        tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

      self.numel = tf.constant([[0]])
    tf.summary.tensor_summary('lat_state',self.z_mu)
    #Define one op to call all summaries
    self.merged = tf.summary.merge_all()
    #and one op to initialize the variables
    self.init_op = tf.global_variables_initializer()

