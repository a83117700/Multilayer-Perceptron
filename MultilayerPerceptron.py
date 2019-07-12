
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter


# In[21]:


def normalize(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0)
    return d

def trans_label(data, n):
    y = (np.array(pd.get_dummies(data[:,n]))[:,:1]).reshape(-1)
    #y[ y == 0] = -1
    data[:,n] = np.float64(np.int8(y))
    return data

def trans_multi_label(data, n, label_num):
    y = np.array(pd.get_dummies(data[:,-1]))
    y = np.float64(np.int8(y))
    data = np.concatenate([data[:,:-1],y],axis = 1)
    return data

def data_processing(raw_data, file_name):
    data = [x.split() for x in raw_data]
    data = [list(map(float,x)) for x in data]
    data = np.array(data)
    label_num = len(np.unique(data[:,-1]))
    
    if(label_num==2):
        data = trans_label(data, data.shape[-1]-1)
    else:
        data = trans_multi_label(data, data.shape[-1]-1,label_num)
    if(file_name == 'IRIS'):
        data = normalize(data)
    return data,label_num

def random_sampling(data, train_percentage, label_num, number=0):
    n = len(data)
    index = int(n*train_percentage)
    np.random.shuffle(data)
    if(n<10):
        if(number==0):
            return data[:,:-1], data[:,-1:], data[:,:-1], data[:,-1:]
        else:
            return data[:,:-label_num], data[:,-label_num:], data[:,:-label_num], data[:,-label_num:]
    elif(label_num!=2):
        return data[:index,:-label_num], data[:index,-label_num:], data[index:,:-label_num], data[index:,-label_num:]
    else:    
        return data[:index,:-1], data[:index,-1:], data[index:,:-1], data[index:,-1:]

def initial_weight(m,n):
    return np.random.normal(size = (m,n)) 

def add_bias(data):
    return np.concatenate((np.ones((data.shape[0], 1))*-1, data), axis = 1)
    
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
    
def delta_sigmoid(X):
    return np.ones((X).shape)-X
    
def output(X, W):
    return sigmoid((X*W).sum(axis = 1)).reshape(-1,1)

def concate_layer_input_matrix(layer_input_matrix, layer_input, neuron):
    shape = layer_input.shape[0]
    try:    
        for i in range(neuron):
            layer_input_matrix = np.concatenate([layer_input_matrix,layer_input], axis = 0)
    except:
        for i in range(neuron-1):
            layer_input_matrix = np.concatenate([layer_input,layer_input], axis = 0)
    return layer_input_matrix.reshape(-1,shape)
    
def feed_forward(layer_input,W,layer,neuron):
    shape = layer_input.shape[0]
    layer_input_matrix = np.array([])

    for i in range(0,layer):
        layer_input_matrix = concate_layer_input_matrix(np.array([]), layer_input, neuron)
        layer_y = output(layer_input,W[i*neuron:(i+1)*neuron])
        layer_input = add_bias(layer_y.reshape(1,-1))

    layer_input_matrix = np.concatenate([layer_input_matrix,layer_input], axis = 0).reshape(-1,shape)

    hat_y = output(layer_input,W[-1])
    return hat_y, layer_input_matrix
    
def gradient(W,theta,X,y,layer,neuron):
    #forward
    layer_input = X
    hat_y, layer_input_matrix = feed_forward(layer_input,W,layer,neuron)
    
    delta_0 = ((y-hat_y)*hat_y*(1-hat_y)).reshape(-1)

    #backward
    layer_delta = delta_sigmoid(layer_input_matrix[layer*neuron][1:])*delta_0*layer_input_matrix[layer*neuron][1:]*W[-1][1:]
    layer_delta_matrix = np.concatenate([layer_delta,delta_0], axis = 0)
    for i in range(layer-1,0,-1):
        layer_delta = delta_sigmoid(layer_input_matrix[i*neuron][1:])*layer_delta*layer_input_matrix[i*neuron][1:]*W[-1][1:]
        layer_delta_matrix = np.concatenate([layer_delta,layer_delta_matrix], axis = 0)

    W = W+(theta*(layer_input_matrix.T)*layer_delta_matrix).T

    return W

def predict(layer_input,W,layer,neuron):
    hat_y, layer_input_matrix = feed_forward(layer_input[0],W,layer,neuron)
    prediction_array = hat_y
    for i in range(1,layer_input.shape[0]):
        hat_y, layer_input_matrix = feed_forward(layer_input[i],W,layer,neuron)
        prediction_array = np.concatenate([prediction_array,hat_y])
    return prediction_array
    
def train(X, y, theta, iters, layer, neuron, early_stop_acc):
    lr_flag = 0
    higest_acc = 0
    neurons_size = layer*neuron+1
    W = initial_weight(neurons_size,X.shape[1])
    for i in range(iters):
        mod_i = i%X.shape[0]
        W = gradient(W,theta,X[mod_i],y[mod_i][0],layer,neuron)
        if(mod_i == 0):
            if(lr_flag==0):
                lr_flag = 1
            else:
                theta = theta*0.95
            recent_acc = performance(predict(X, W,layer,neuron), y)[0]
            if(recent_acc==1):
                break
            elif(recent_acc>higest_acc):
                higest_W = W
                higest_acc = recent_acc
    if(performance(predict(X, W,layer,neuron), y)[0]<higest_acc):
        W = higest_W
    return W

def performance(hat_y_array, y):
    n = len(y)
    rmse = np.sqrt(((y-hat_y_array)**2).sum()/n)
    hat_y_round_array = np.round(hat_y_array)
    
    p = 0
    for i in range(n):
        if(hat_y_round_array[i] == y[i]):
            p = p+1 
    return p/n,rmse

def read_file(file_name):
    file_address = 'DataSet\\'+file_name+'.txt'
    with open(file_address) as f:
        raw_data = f.readlines()
    
    if(file_name=='Number'):
        number=1
    else:
        number=0
    data,label_num = data_processing(raw_data, file_name)
    data = add_bias(data)
    
    train_x, train_y, test_x, test_y = random_sampling(data, 2/3, label_num, number)
    return train_x, train_y, test_x, test_y, label_num
        

def perceptron(train_x, train_y, test_x, test_y, theta, iters, early_stop_acc, layer, neuron):
    W = train(train_x, train_y, theta, iters, layer, neuron, early_stop_acc)
    acc_train, rmse_train = performance(predict(train_x, W, layer, neuron), train_y)
    acc_test, rmse_test = performance(predict(test_x, W, layer, neuron), test_y)
    
    return acc_train, rmse_train, acc_test, rmse_test, W


def split_pos_neg(X,y,n):
    row_p_idx = np.where(y == 1)[0]
    row_n_idx = np.where(y == 0)[0]
    col_idx = np.array(list(range(n+1)))
    X_p = X[row_p_idx[:, None], col_idx]
    X_n = X[row_n_idx[:, None], col_idx]
    return X_p, X_n


# In[24]:


from tkinter import *
from tkinter.ttk import *
global train_x, train_y, test_x, test_y, acc_train, acc_test, W, label_num, file_name


def clicked_train():
    global train_x, train_y, test_x, test_y, acc_train, acc_test, W, label_num, file_name
    
    lr = float(input_lr.get())
    iters = int(float(input_iters.get()))
    early_stop_acc = float(input_AccThreshold.get())
    layer = 1
    #neuron = label_num
    neuron = train_x.shape[1]-1
    if(label_num==2):
        acc_train, rmse_train, acc_test, rmse_test, W = perceptron(train_x, train_y, test_x, test_y, lr, iters, early_stop_acc, layer, neuron)
        prediction_train_list = predict(train_x, W, layer, neuron)
        prediction_test_list = predict(train_x, W, layer, neuron)
    else:
        prediction_train_list = np.array([])
        prediction_test_list = np.array([])
        for i in range(label_num):
            acc_train, rmse_train, acc_test, rmse_test, W = perceptron(train_x, train_y[:,i:i+1], test_x, test_y[:,i:i+1], lr, iters, early_stop_acc, layer, neuron)
            prediction_train_list = np.append(prediction_train_list, predict(train_x, W, layer, neuron))
            prediction_test_list = np.append(prediction_test_list, predict(test_x, W, layer, neuron))
        prediction_train_list = prediction_train_list.reshape(-1,len(train_x)).T
        prediction_test_list = prediction_test_list.reshape(-1,len(test_x)).T
        acc_train, rmse_train = performance(train_y.argmax(axis = 1).reshape(-1,1),prediction_train_list.argmax(axis = 1).reshape(-1,1))
        acc_test, rmse_test = performance(test_y.argmax(axis = 1).reshape(-1,1),prediction_test_list.argmax(axis = 1).reshape(-1,1))
    
    train_p_x, train_n_x = split_pos_neg(train_x, train_y, 2)
    test_p_x, test_n_x = split_pos_neg(test_x, test_y, 2)
    

    
    fig1 = Figure(figsize=(5,5))
    train_fig = fig1.add_subplot(111)
    
    fig2 = Figure(figsize=(5,5))
    test_fig = fig2.add_subplot(111)
    
    if(label_num==2):
        plot_classify_line(fig1, train_fig, train_x, train_y, train_p_x, train_n_x, 0, W, layer, neuron)
        plot_classify_line(fig2, test_fig, test_x, test_y, test_p_x, test_n_x, 4, W, layer, neuron)
        
    if(file_name!='Number'):
        var_acc_train.set('train accuracy : '+str(round(acc_train,4))+', train RMSE : '+str(round(rmse_train,4)))
        var_acc_test.set('test accuracy : '+str(round(acc_test,4))+', test RMSE : '+str(round(rmse_test,4)))
    else:
        var_acc_train.set('accuracy : '+str(round(acc_train,4))+', RMSE : '+str(round(rmse_train,4)))
        var_acc_test.set('Prediction : '+str(prediction_train_list.argmax(axis = 1).reshape(-1,)))
    #var_w.set('鍵結值\n W0:'+str(W[0])+'\n W1:'+str(W[1])+'\n W2:'+str(W[2]))
    
    
    lbl_train = Label(window, textvariable=var_acc_train) 
    lbl_train.grid(column=1, row=4)

    lbl_test = Label(window, textvariable=var_acc_test) 
    lbl_test.grid(column=4, row=4)
    
    lbl_w = Label(window, textvariable=var_w) 
    lbl_w.grid(column=2, row=5, columnspan=2)
    
    
def plot_classify_line(fig, sub_fig, X, y, p_x, n_x, col, W, layer, neuron):
    
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))

    # Put the result into a color plot
    Z = np.round(predict(np.c_[-1*np.ones(xx.shape).ravel(),xx.ravel(), yy.ravel()], W, layer, neuron).reshape(xx.shape))
    
    #plt.axis('off')

    # Plot also the training points
    #plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)
    
    sub_fig.scatter(p_x[:,1:2],p_x[:,2:3], color = 'r')
    sub_fig.scatter(n_x[:,1:2],n_x[:,2:3], color = 'b')
    sub_fig.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.5)
    
    axes = fig.gca()
    x_range = np.array(axes.get_xlim())
    y_range = np.array(axes.get_ylim())
    axes.set_xlim([x_range[0],x_range[1]])
    axes.set_ylim([y_range[0],y_range[1]])

    #abline(fig, sub_fig, slope, intercept)
    a = fig.gca()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(column=col, row=3, columnspan=3)
    canvas.show()   

def plot_data_in_tk(fig, sub_fig, p_x, n_x, col):
    sub_fig.scatter(p_x[:,1:2],p_x[:,2:3], color = 'r')
    sub_fig.scatter(n_x[:,1:2],n_x[:,2:3], color = 'b')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(column=col, row=3, columnspan=3)
    canvas.show()
    
def plot_number(fig, sub_fig, data, col):
    sub_fig.imshow(data.reshape(5,5), cmap='gray')
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(column=col, row=3, columnspan=3)
    canvas.show()
    
def combobox_selected(eventObject):
    global train_x, train_y, test_x, test_y, label_num, file_name

    file_name = combo_data.get()
    
    train_x, train_y, test_x, test_y,label_num = read_file(file_name)
    train_p_x, train_n_x = split_pos_neg(train_x, train_y, 2)
    test_p_x, test_n_x = split_pos_neg(test_x, test_y, 2)
    if(file_name=='Number'):
        fig1 = Figure(figsize=(5,5))
        zero_fig = fig1.add_subplot(111)
        plot_number(fig1, zero_fig, train_x[0,1:], 0)
        fig2 = Figure(figsize=(5,5))
        one_fig = fig2.add_subplot(111)
        plot_number(fig2, one_fig, train_x[1,1:], 3)
        fig3 = Figure(figsize=(5,5))
        two_fig = fig3.add_subplot(111)
        plot_number(fig3, two_fig, train_x[2,1:], 6)
        fig4 = Figure(figsize=(5,5))
        three_fig = fig4.add_subplot(111)
        plot_number(fig4, three_fig, train_x[3,1:], 9)

    else:
        fig1 = Figure(figsize=(5,5))
        train_fig = fig1.add_subplot(111)
        plot_data_in_tk(fig1, train_fig, train_p_x, train_n_x, 0)
        fig2 = Figure(figsize=(5,5))
        test_fig = fig2.add_subplot(111)
        plot_data_in_tk(fig2, test_fig, test_p_x, test_n_x, 4)
    

window = Tk()

window.title("Perceptron")
window.geometry('1800x800')

var_acc_train = StringVar()
var_acc_test = StringVar()
var_w = StringVar()

lbl_data = Label(window, text="Select Dataset")
lbl_data.grid(column=0, row=0)
combo_data = Combobox(window, width=10)
combo_data['values']= ('perceptron1','perceptron2','2Ccircle1','2Circle1','2Circle2','2CloseS','2CloseS2','2CloseS3','2cring','2CS','2Hcircle1','2ring',
                       '4satellite-6','5CloseS1','8OX','C3D','C10D','IRIS','Number','perceptron3','perceptron4','wine','xor')
combo_data.grid(column=1, row=0)
combo_data.bind("<<ComboboxSelected>>", combobox_selected)


lbl_lr = Label(window, text="Input learning rate")
lbl_lr.grid(column=0, row=1, pady = 5, padx = 5)
input_lr = Entry(window,width=10)
input_lr.insert(END, '0.8')
input_lr.grid(column=1, row=1)

lbl_iters = Label(window, text="Input iterations")
lbl_iters.grid(column=2, row=1, pady = 5, padx = 5)
input_iters = Entry(window,width=10)
input_iters.insert(END, '1000')
input_iters.grid(column=3, row=1)

lbl_AccThreshold = Label(window, text="Input accuracy threshold to early stop")
lbl_AccThreshold.grid(column=4, row=1, pady = 5, padx = 5)
input_AccThreshold = Entry(window,width=10)
input_AccThreshold.insert(END, '1')
input_AccThreshold.grid(column=5, row=1)

train_btn = Button(window, text="Start training", command = clicked_train)
train_btn.grid(column=0, row=2)

window.mainloop()

