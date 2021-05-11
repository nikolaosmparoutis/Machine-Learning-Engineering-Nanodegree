import torch
import torch.nn as nn
import torch.nn.functional as F

## TODO: Complete this classifier
class SimpleNet(nn.Module):
    
    ## TODO: Define the init function
    def __init__(self, input_dim, hidden_dim, output_dim):
        '''Defines layers of a neural network.
           :param input_dim: Number of input features
           :param hidden_dim: Size of hidden layer(s)
           :param output_dim: Number of outputs
         '''
        super(SimpleNet, self).__init__()

        #         From my past experience i use this formula to find the optimal number of nodes
        #         Nh=Ns(α∗(Ni+No)) 

        #         Ni = number of input neurons.
        #         No = number of output neurons.
        #         Ns = number of samples in training data set.
        #         α = an arbitrary scaling factor usually 2-10.
        
        #         Nh = ... = [22,75]
        
        #  Define all layers, here. I choose 3+1 layers.
        
        print("Default numbers for, input_dim, hidden_dim, output_dim", input_dim, hidden_dim, output_dim)
        hid_layer1 = hidden_dim
        hid_layer2 = hidden_dim
        
        self.fc1 = nn.Linear(input_dim, hid_layer1)
        self.bn1 = nn.LayerNorm(hid_layer1)
        self.fc2 = nn.Linear(hid_layer1, hid_layer2)
        self.bn2 = nn.LayerNorm(hid_layer2)
        self.fc3 = nn.Linear(hid_layer2, output_dim)
        # dropout prevents overfitting of data
        self.drop = nn.Dropout(0.3)
        
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        '''Feedforward behavior of the net.
           :param x: A batch of input features
           :return: A single, sigmoid activated value
         '''
        x = self.fc1(x)
        x = self.bn1(x) # usually added before ReLU(as mentioned in the Batch Normalization paper), but not is not standard. 
        x = F.leaky_relu(x, negative_slope=0.01)  # prevent dying relu and zeroing the gradient
        
        x = self.drop(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        
        x = self.drop(x)
        x = self.fc3(x)
        x = F.leaky_relu(x, negative_slope=0.01)
        
        return x