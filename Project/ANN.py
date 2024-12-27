import numpy as np
from copy import deepcopy

#functions of non-linear activations
def f_sigmoid(X, deriv=False):
    if not deriv:
        # Numerically stable sigmoid definition. courtesy of https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
        return np.where(
            X >= 0, # condition
            1 / (1 + np.exp(-X)), # For positive values
            np.exp(X) / (1 + np.exp(X)) # For negative values
        )
    else:
        return f_sigmoid(X)*(1 - f_sigmoid(X))


def f_softmax(X):
    Z = np.sum(np.exp(X), axis=1)
    Z = Z.reshape(Z.shape[0], 1)
    return np.exp(X) / Z

#Functionality of a single hidden layer
class Layer:
    def __init__(self, size, batch_size, is_input=False, is_output=False,
                 activation=f_sigmoid, sigma=1E-4, rng=None):
        
        if rng is None:
            rng = np.random.default_rng()

        self.is_input = is_input
        self.is_output = is_output

        # Z is the matrix that holds output values
        self.Z = np.zeros((batch_size, size[0]))
        # The activation function is an externally defined function (with a
        # derivative) that is stored here
        self.activation = activation

        # W is the outgoing weight matrix for this layer
        self.W = None
        # S is the matrix that holds the inputs to this layer
        self.S = None
        # D is the matrix that holds the deltas for this layer
        self.D = None
        # Fp is the matrix that holds the derivatives of the activation function
        self.Fp = None

        if not is_input:
            self.S = np.zeros((batch_size, size[0]))
            self.D = np.zeros((batch_size, size[0]))

        if not is_output:
            self.W = rng.normal(size=size, scale=sigma)

        if not is_input and not is_output:
            self.Fp = np.zeros((size[0], batch_size))

    def forward_propagate(self):
        if self.is_input:
            return self.Z.dot(self.W)

        self.Z = self.activation(self.S)
        if self.is_output:
            return self.Z
        else:
            # For hidden layers, we add the bias values here
            self.Z = np.append(self.Z, np.ones((self.Z.shape[0], 1)), axis=1)
            self.Fp = self.activation(self.S, deriv=True).T
            return self.Z.dot(self.W)
        

class MultiLayerPerceptron:
    def __init__(self, layer_config, batch_size=100, activation=f_sigmoid, sigma=1E-4, seed = 0):
        self.layers = []
        self.num_layers = len(layer_config)
        self.minibatch_size = batch_size

        rng = np.random.default_rng(seed=seed) 

        for i in range(self.num_layers-1):
            if i == 0:
                #print ("Initializing input layer with size {0}.".format(layer_config[i]))
                # Here, we add an additional unit at the input for the bias
                # weight.
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                         batch_size,
                                         is_input=True,
                                         activation=activation,
                                         sigma=sigma,
                                         rng=rng))
            else:
                #print ("Initializing hidden layer with size {0}.".format(layer_config[i]))
                # Here we add an additional unit in the hidden layers for the
                # bias weight.
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                         batch_size,
                                         activation=activation,
                                         sigma=sigma,
                                         rng=rng))

        #print ("Initializing output layer with size {0}.".format(layer_config[-1]))
        self.layers.append(Layer([layer_config[-1], None],
                                 batch_size,
                                 is_output=True,
                                 activation=f_softmax,
                                 rng=rng))
        #print ("Done!")

    def forward_propagate(self, data):
        # We need to be sure to add bias values to the input
        self.layers[0].Z = np.append(data, np.ones((data.shape[0], 1)), axis=1)

        for i in range(self.num_layers-1):
            self.layers[i+1].S = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, yhat, labels):
        
        #exit_with_err("FIND ME IN THE CODE, What is computed in the next line of code?\n")
        # The derivative of the loss with respect to the logits with a softmax/cross-entropy function.

        self.layers[-1].D = (yhat - labels).T
        for i in range(self.num_layers-2, 0, -1):
            # We do not calculate deltas for the bias values
            W_nobias = self.layers[i].W[0:-1, :]
            
            #exit_with_err("FIND ME IN THE CODE, What does this 'for' loop do?\n")
            # Calculates to derivative of each layer i based on the derivative of layer i+1 and the activation function for all layers in the ANN.
            
            self.layers[i].D = W_nobias.dot(self.layers[i+1].D) * self.layers[i].Fp

    def update_weights(self, eta):
        for i in range(0, self.num_layers-1):
            W_grad = -eta*(self.layers[i+1].D.dot(self.layers[i].Z)).T
            self.layers[i].W += W_grad

    def evaluate(self, train_data, train_labels, test_data, test_labels,
                 num_epochs=70, eta=0.05, eval_train=False, eval_test=True):

        N_train = len(train_labels)*len(train_labels[0])
        N_test = len(test_labels)*len(test_labels[0])

        #print ("Training for {0} epochs...".format(num_epochs))
        max_acc = 0
        best_epoch = 0
        best_model = None

        for t in range(0, num_epochs):
            out_str = "[{0:4d}] ".format(t)

            for b_data, b_labels in zip(train_data, train_labels):
                output = self.forward_propagate(b_data)
                self.backpropagate(output, b_labels)
                
                #exit_with_err("FIND ME IN THE CODE, How does weight update is implemented? What is eta?\n")
                # The weight updateing is done using the learning rule w_new = w - learning_rate*dJ/dw, eta is the learning rate.

                self.update_weights(eta=eta)

            if eval_train:
                errs = 0
                for b_data, b_labels in zip(train_data, train_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = ("{0} Training error: {1:.5f}".format(out_str,
                                                           float(errs)/N_train))

            if eval_test:
                errs = 0
                for b_data, b_labels in zip(test_data, test_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = ("{0} Val error: {1:.5f}").format(out_str,
                                                       float(errs)/N_test)
                
            test_accuracy = 0
            for b_data, b_labels in zip(test_data, test_labels):
                output = self.forward_propagate(b_data)
                yhat = np.argmax(output, axis=1)
                test_accuracy += np.sum(b_labels[np.arange(len(b_labels)), yhat])
            test_accuracy = float(test_accuracy)/N_test

            if test_accuracy>max_acc:
                max_acc = test_accuracy
                best_model = deepcopy(self)
                best_epoch = t

        return best_model

    def get_accuracy(self, test_data, test_labels):

        N_test = len(test_labels)*len(test_labels[0])
                
        test_accuracy = 0
        for b_data, b_labels in zip(test_data, test_labels):
            output = self.forward_propagate(b_data)
            yhat = np.argmax(output, axis=1)
            test_accuracy += np.sum(b_labels[np.arange(len(b_labels)), yhat])
        test_accuracy = float(test_accuracy)/N_test

        return test_accuracy

def label_to_bit_vector(labels, nbits):
    bit_vector = np.zeros((labels.shape[0], nbits))
    for i in range(labels.shape[0]):
        bit_vector[i, labels[i]] = 1.0

    return bit_vector

def create_batches(data, labels, batch_size, create_bit_vector=False, nclasses = 10):
    N = data.shape[0]

    chunked_data = []
    chunked_labels = []
    idx = 0
    while idx + batch_size <= N:
        chunked_data.append(data[idx:idx+batch_size, :])
        if not create_bit_vector:
            chunked_labels.append(labels[idx:idx+batch_size])
        else:
            bit_vector = label_to_bit_vector(labels[idx:idx+batch_size], nclasses)
            chunked_labels.append(bit_vector)

        idx += batch_size

    return chunked_data, chunked_labels

def prepare_for_backprop(batch_size, Train_images, Train_labels, Valid_images, Valid_labels, nclasses=10):
    
    batched_train_data, batched_train_labels = create_batches(Train_images, Train_labels,
                                              batch_size,
                                              create_bit_vector=True,
                                              nclasses=nclasses)
    batched_valid_data, batched_valid_labels = create_batches(Valid_images, Valid_labels,
                                              batch_size,
                                              create_bit_vector=True,
                                              nclasses=nclasses)


    return batched_train_data, batched_train_labels,  batched_valid_data, batched_valid_labels