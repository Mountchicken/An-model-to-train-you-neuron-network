import numpy as np
import matplotlib.pyplot as plt
from dnn_utils import *
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters={}
    L=len(layer_dims)
    for i in range(1,L):
        parameters['W'+str(i)]=np.random.randn(layer_dims[i],layer_dims[i-1])*np.sqrt(2 / layer_dims[i - 1])
        parameters['b'+str(i)]=np.zeros((layer_dims[i],1))
    return parameters

def single_forward_propagation(A_prev,W,b,activation,keep_prob=1,ifDrop_out=False):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing A_prev,W,Z
    """

    
    Z=np.dot(W,A_prev)+b
    if activation=="sigmoid":
        A=sigmoid(Z)

    if activation=="relu":
        A=relu(Z)
    '''Dropout'''
    if ifDrop_out:
        D=np.random.rand(A.shape[0],A.shape[1])
        D=D<keep_prob
        A=(A*D)/keep_prob
        cache={"A_prev":A_prev,"W":W,"Z":Z,"D":D}
    else :
        cache={"A_prev":A_prev,"W":W,"Z":Z}
    return A,cache

def L_model_forward(X,parameters,keep_prob=1,ifDrop_out=False):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    L=len(parameters)//2
    caches=[]
    A=X
    '''先执行L-1层前面的relu传播'''
    for i in range(1,L):
        W=parameters['W'+str(i)]
        b=parameters['b'+str(i)]
        A,cache=single_forward_propagation(A,W,b,'relu')
        caches.append(cache)
    '''再执行最后一层的sigmoid传播'''
    W=parameters['W'+str(L)]
    b=parameters['b'+str(L)]
    AL,cache=single_forward_propagation(A,W,b,'sigmoid')
    caches.append(cache)
    return AL,caches

def compute_cost(AL,Y,caches,lambd):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
    caches -- cache of W, b, A_prev of each layer, which we will only use W to compute the L2_regularization_cost
    lambd -- parameters for L2_regularization_cost
    Returns:
    cost -- cross-entropy cost
    """
    m=Y.shape[1]
    cost = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    '''新增L2正则化项'''
    sump=0
    for lst in caches:
        sump=sump+np.sum(np.square(lst['W']))
    L2_regularization_cost=lambd*sump/(2*m)
    '''新增L2正则化项'''
    cost = np.squeeze(cost+L2_regularization_cost)    # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost

def single_backward_propagation(dA,cache,activation,lambd,keep_prob=1,ifDrop_out=False):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache --A_prev,W,Z
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev=cache["A_prev"]
    W=cache["W"]
    Z=cache["Z"]
    if ifDrop_out:
        D=cache["D"]
        dA=(dA*D)/keep_prob
    #(A_prev,W,b),Z=cache
    m=A_prev.shape[1]
    if activation=='relu':
        dZ=relu_backward(dA,Z)
    if activation=='sigmoid':
        dZ=sigmoid_backward(dA,Z)
    '''新增L2正则化项'''
    dW=np.dot(dZ,A_prev.T)/m+lambd*W/m
    '''新增L2正则化项'''
    db=np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev=np.dot(W.T,dZ)
    cache={"dW":dW,"db":db}
    return dA_prev,cache

def L_model_backward(AL,Y,caches,lambd,keep_prob=1,ifDrop_out=False):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads={}
    L=len(caches)
    Y=Y.reshape(AL.shape)
    '''Initializing the backpropagation'''
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    '''首先是最后一层的sigmoid'''
    dA_prev,cache=single_backward_propagation(dAL,caches[L-1],'sigmoid',lambd,keep_prob=1,ifDrop_out=False)
    grads["db"+str(L)]=cache["db"]
    grads["dW"+str(L)]=cache["dW"]
    '''接下来是L-1层的relu'''
    for i in reversed(range(L-1)):
        dA_prev,cache=single_backward_propagation(dA_prev,caches[i],'relu',lambd,keep_prob=1,ifDrop_out=False)
        grads["db"+str(i+1)]=cache["db"]
        grads["dW"+str(i+1)]=cache["dW"]
    
    return grads

def update_parameters(parameters,grads,learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    L=len(parameters)//2
    for i in range(L):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * grads["dW" + str(i+ 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]
    return parameters

def predict(X,Y,parameters):
    """
    该函数用于预测L层神经网络的结果，当然也包含两层
    参数：
     X - 测试集
     y - 标签
     parameters - 训练模型的参数
    返回：
     p - 给定数据集X的预测
    """
    m = X.shape[1]
    n = len(parameters) // 2 # 神经网络的层数
    p = np.zeros((1,m)) 
    #根据参数前向传播
    probas, caches = L_model_forward(X, parameters)
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("准确度为: {} ".format(float(np.sum((p == Y))/m)))
    return p

def model(X,Y,layer_dims,learning_rate=0.0075,lambd=0,num_iterations=3000,keep_prob=1,ifDrop_out=False,print_cost=False,plot_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    costs=[]
    parameters=initialize_parameters_deep(layer_dims)
    for i in range(num_iterations):
        '''forward'''
        AL,caches=L_model_forward(X,parameters,keep_prob=1,ifDrop_out=False)
        '''cost'''
        cost=compute_cost(AL,Y,caches,lambd)
        '''backward'''
        grads=L_model_backward(AL,Y,caches,lambd,keep_prob=1,ifDrop_out=False)
        '''update'''
        parameters=update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)
    # plot the cost
    if plot_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return parameters