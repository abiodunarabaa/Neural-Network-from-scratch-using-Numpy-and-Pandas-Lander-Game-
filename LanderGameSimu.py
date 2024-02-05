import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy.random as rnd
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\Users\840-G2\Downloads\ce889_dataCollection.csv")  #importation of dataset

df_updated = df.drop_duplicates() #To regularize possible duplicated rows

df.plot (y= ["x_dist", "y_dist", "xvel", "yvel" ], layout = (4,2), subplots =  True, kind= "box", figsize=(15,15) ) #To identify possible outliers/incomplete data entries

df.skew()

df_array = np.asarray(df) #the conversion of the dataset to array

dfMin = np.min(df_array[:,:], axis= 0) #the minimum values in each column
dfMax = np.max(df_array[:,:], axis= 0 ) #the maximum values in each column
dfNorm = (df_array - dfMin) / (dfMax - dfMin) #normalization of the dataset


dfNorm_xDist = dfNorm[:,0]
dfNorm_yDist = dfNorm[:,1]
input_data = np.column_stack((dfNorm_xDist, dfNorm_yDist)) #the orocessed input data


dfNorm_xVel = dfNorm[:,2]
dfNorm_yVel = dfNorm[:,3]
output_data = np.column_stack((dfNorm_xVel, dfNorm_yVel)) #the processed output data

x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size= 0.2, random_state=2) #the slipting of normalized data to train and test partitions
print("Total input:", input_data.shape)
print("Total output:", output_data.shape)
print("X_train:", x_train.shape)
print("X_test:", x_test.shape)

X = x_test 
y = y_test



net_architecture = [2,3,2] #Topology/Architecture

LR  = 0.15 #Learning rate

mm = 0.25 #momentum

epochs = 800

def func_sigmoid(x): #Sigmoid activation deployed to ensure the input node is activated i.e between range 0 and 1
    return 1/(1+np.exp(-x))
              
def func_deriv(x):
    return func_sigmoid(x)*(1- func_sigmoid(x))

def weight_init(net_architecture): #weight and bias
    W = {}      #weight
    b = {}      #bias
    for l in range(1, len(net_architecture)):
        W[l] = rnd.random_sample((net_architecture[l], net_architecture[l-1]))
        b[l] = rnd.random_sample((net_architecture[l]))
    return W,b

def init_values(net_architecture): #initialize the weight and bias
    dv_W = {}
    dv_b = {}
    for l in range(1, len(net_architecture)):
        dv_W[l] = np.zeros((net_architecture[l], net_architecture[l-1]))
        dv_b[l] = np.zeros((net_architecture[l]))
    return dv_W, dv_b

def feed_forward(x,W,b): #Feed forward function with 3 arguments i.e input neuron, weight and bias
    h = {1:x}
    z = {}
    for l in range(1, len(W) + 1):
        if l == 1:
            input_neuron = x
        else:
            input_neuron = h[l]
        z[l+1] = W[l].dot(input_neuron) +b[l]
        h[l+1] = func_sigmoid(z[l+1])
    return h,z

def backpropagate(y,h_out,z_out): #The backpropagation on the outlayer i.e subtractiing the output from the expected output
    return -(y-h_out) * func_deriv(z_out)


def der_backpropa(delta_plus_l, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_plus_l) * func_deriv(z_l)


def net_training(net_architecture, X, y): #Training of the network with input and output/target data
    W,b = weight_init(net_architecture)
    counter = 0
    num_count = len(y)
    rmse = []
    print('The gradient descent for {} epochs'.format(epochs))
    while counter < epochs:
        if counter%100 == 0:
            print('Epoch {} of {}'.format(counter, epochs))

        dv_W, dv_b = init_values(net_architecture)
        rms = 0
        for i in range(len(y)):
            delta = {}
            h,z = feed_forward(X[i,:], W, b)
            for l in range(len(net_architecture), 0, -1):
                if l == len(net_architecture):
                    delta[l] = backpropagate(y[i,:], h[l], z[l])
                    rms = rms + np.linalg.norm((y[i,:] -h[l]))
                else:
                    if l > 1:
                        delta[l] = der_backpropa(delta[l+1], W[l], z[l])
                        dv_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis]))
        for l in range(len(net_architecture) - 1, 0, -1):
            W[l] = ( W[l]) - (LR * (1.0/num_count * dv_W[l]) )  
            b[l] = ( b[l]) - (LR * (1.0/num_count * dv_b[l]) )

            rms = (1.0/num_count * rms**2)      #Root Square Mean Error evaluation
            rms_comp = math.sqrt(rms)
            rmse.append(rms_comp)
            counter = counter + 1
    
    print ("W: ", W)
    #print ("b: ", b)
    print ("RMSE", rmse )
    #print (X)
    #print (y)
    return W,b,rmse

W,b,rmse = net_training(net_architecture, x_train, y_train)


plt.plot(rmse) #displaying of RMSE graph
plt.ylabel('RMSE')
plt.xlabel('Epochs')
plt.show()




