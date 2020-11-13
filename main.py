import numpy as np
import h5py
import matplotlib.pyplot as plt
from lr_utils import *
from dnn_model import model,predict
''' load dataset '''
train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes =load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T 
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y




layers_dims = [12288,20,7,5,1] #  5-layer model
parameters =model(train_x, train_y, layers_dims,lambd=0,keep_prob=0.84,ifDrop_out=True, num_iterations = 1500, print_cost = True,learning_rate=0.0075)
predictions_train = predict(train_x, train_y, parameters) #训练集
predictions_test = predict(test_x, test_y, parameters) #测试集

