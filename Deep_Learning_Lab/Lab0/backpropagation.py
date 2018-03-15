import numpy as np
import math
class DNN:
    # hidden_layers = [2] for one hidden layer with two neuron
    # hidden_layers = [2, 2] for two hidden layers with two neuron
    # hidden_layers = [2, 2, 2] for three hidden layers with two neuron
    def __init__(self, X, Y, learning_rate = 0.5, num_input_neuron = 2, hidden_layers = [2, 2], num_output_neuron = 1):  
        # set model
        self.num_hidden_layer = len(hidden_layers)
        self.num_input_neuron = num_input_neuron
        self.hidden_layers = hidden_layers
        self.num_output_neuron = num_output_neuron
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate

        # create and initial weights 
        self.weights = []
        # input to first hidden layer  
        self.weights.append(np.random.uniform(-0.2, 0.2, [self.num_input_neuron, self.hidden_layers[0]]))
        # hidden layer to hidden layer
        for i in range(self.num_hidden_layer-1):
            self.weights.append(np.random.uniform(-0.2, 0.2, [self.hidden_layers[i], self.hidden_layers[i+1]]))
        # hidden layer to output layer 
        self.weights.append(np.random.uniform(-2.0, 2.0, [self.hidden_layers[-1], self.num_output_neuron]))
        
        # create and initial bias
        self.bias = [1.0]*(self.num_hidden_layer+1)
        # create weights for bias
        self.bias_weights = []
        # input to first hidden layer 
        self.bias_weights.append(np.random.normal(0.0, 1.0, [self.hidden_layers[0]]))
        # hidden layer to hidden layer 
        for i in range(self.num_hidden_layer-1):
            self.bias_weights.append(np.random.normal(0.0, 1.0, [self.hidden_layers[i+1]]))
        # hidden layer to output layer
        self.bias_weights.append(np.random.normal(0.0, 1.0, [self.num_output_neuron]))

        # create neuron outputs
        self.outputs = []
        # input layer
        self.outputs.append([0.0]*self.num_input_neuron)
        # hidden layer to hidden layer
        for i in range(self.num_hidden_layer):
            self.outputs.append([0.0]*self.hidden_layers[i])
        # output layer
        self.outputs.append([0.0]*self.num_output_neuron)

    def backward(self, target):
        
        # output layer to hidden layer
        # delta out for output layer
        delta_out = [0.0]*self.num_output_neuron
        for i in range(self.num_output_neuron):
            delta_out[i] = (self.outputs[-1][i] - target)*dsigmoid(self.outputs[-1][i])
        # change weights between output layer and hidden layer
        for i in range(self.hidden_layers[-1]):
            for j in range(self.num_output_neuron):
                self.weights[-1][i][j] -= self.learning_rate*delta_out[j]*self.outputs[-2][i]
        
       
        # delta out for last hidden layer
        delta_hidden = []
        for i in range(self.num_hidden_layer):
            delta_hidden.append([0.0]*self.hidden_layers[i])
        for i in range(self.hidden_layers[-1]):
            for j in range(self.num_output_neuron):
                delta_hidden[-1][i] += delta_out[j]*self.weights[-1][i][j]
            delta_hidden[-1][i] *= dsigmoid(self.outputs[-2][i])
        
        # hidden layer to hidden layer
        for i in range(self.num_hidden_layer-1,0,-1):
            # change weights between hidden layer and hidden layer 
            for j in range(self.hidden_layers[i-1]):
                for k in range(self.hidden_layers[i]):
                    self.weights[i][j][k] -= self.learning_rate*delta_hidden[i][k]*self.outputs[i][j]
            # delta hidden for hidden layers
            for j in range(self.hidden_layers[i-1]):
                for k in range(self.hidden_layers[i]):
                    delta_hidden[i-1][j] += delta_hidden[i][k]*self.weights[i][j][k]
                delta_hidden[i-1][j] *= dsigmoid(self.outputs[i][j])
       
        # change weights between input layer and first hidden layer
        for i in range(self.num_input_neuron):
            for j in range(self.hidden_layers[0]):
                self.weights[0][i][j] -= self.learning_rate*delta_hidden[0][j]*self.outputs[0][i]

        # delta out for bias
        delta_bias_out = delta_out
        # change bias weights between output layer and hidden layer
        for i in range(self.num_output_neuron):
            self.bias_weights[-1][i] -= self.learning_rate*delta_bias_out[i]

        # create hidden layer bias delta
        delta_bias_hidden = [0.0]*self.num_hidden_layer
        # bias delta for last hidden layer
        for j in range(self.num_output_neuron):
            delta_bias_hidden[-1] += delta_bias_out[j]*self.bias_weights[-1][j]
        delta_bias_hidden[-1] *= dsigmoid(self.bias[-1])

        # bias hidden layer to hidden layer
        for i in range(self.num_hidden_layer-1,0,-1):
            # change bias weights between hidden layer and hidden layer
            for k in range(self.hidden_layers[i]):
                self.bias_weights[i][k] -= self.learning_rate*delta_bias_hidden[i]
            # bias delta for hidden layer
            for k in range(self.hidden_layers[i]):
                delta_bias_hidden[i-1] += delta_bias_hidden[i]*self.bias_weights[i][k] ### i or k
            delta_bias_hidden[i-1] *= dsigmoid(self.bias[i])

        # change bias weights between input layer and first hidden layer
        for j in range(self.hidden_layers[0]):
            self.bias_weights[0][j] -= self.learning_rate*delta_bias_hidden[0]
    
    def test(self):
        for i in range(4):
            print ("Inputs: [%d,%d] Output: %.9f Target: %d" % (self.X[i][0],self.X[i][1],self.forward(self.X[i])[0],self.Y[i]))

    def forward(self, data):

        for i in range(self.num_input_neuron):
            self.outputs[0][i] = data[i]

        for i in range(self.hidden_layers[0]):
            sum = 0.0
            for j in range(self.num_input_neuron):
                 sum += self.weights[0][j][i]*self.outputs[0][j]
            self.outputs[1][i] = sigmoid(sum+self.bias[0]*self.bias_weights[0][i])

        for i in range(self.num_hidden_layer-1):
            for j in range(self.hidden_layers[i+1]):
                sum = 0.0
                for k in range(self.hidden_layers[i]):
                     sum += self.weights[i+1][k][j]*self.outputs[i+1][k]
                self.outputs[i+2][j] = sigmoid(sum+self.bias[i]*self.bias_weights[i+1][j])
        
        for i in range(self.num_output_neuron):
            sum =0.0
            for j in range(self.hidden_layers[-1]):
                sum += self.weights[-1][j][i]*self.outputs[-2][j]
            self.outputs[-1][i] = sigmoid(sum+self.bias[-1]*self.bias_weights[-1][i])
        
        return self.outputs[-1]

    def trainNetwork(self):
        for i in range(500001):
            for j in range(4):
                self.forward(self.X[j])
                self.backward(self.Y[j])
            if i%1000 == 0:
                print ('Epoch: ', i)
                self.test()
    
def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def dsigmoid(x):
    return x*(1.0-x)

def main():
    X = [[0,0],
         [0,1],
         [1,0],
         [1,1]]
    Y = [0, 1, 1, 0]

    network = DNN(X,Y)
    network.trainNetwork()

if __name__ == '__main__':
    main()
