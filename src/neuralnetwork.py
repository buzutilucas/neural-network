# coding: utf-8
"""
Created on Mon Jan 05 2018
Python 3.6

@author: Lucas
"""
import numpy as np
import pandas as pd
import os
import csv
from sklearn import datasets

class NeuralNetworkIris:
    """
    Neural Network with three hidden layer
    Activation Function: Hyperbolic.
    Error: Simple Error.
    Data base: Using sklearn and the datasets used, was iris.
    """
    def __init__(self, neuronHiddenLayer0=8, neuronHiddenLayer1=16, neuronHiddenLayer2=4,
                 training_time=1000000, learning_rate=0.0001, momentum=1, Err=0):
        self.neuronHiddenLayer0 = neuronHiddenLayer0
        self.neuronHiddenLayer1 = neuronHiddenLayer1
        self.neuronHiddenLayer2 = neuronHiddenLayer2
        self.training_time = training_time
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.Err = Err
        self.err = 100
        self.synapse = {}
        self.opl = 0


    def hyperbolic(self, add):
        """
        tanh(x) = sinh(x) / cosh(x)

        :param add: x
        :return: Result of tanh x
        """
        return (np.exp(add) - np.exp(-add)) / (np.exp(add) + np.exp(-add))


    def hyperbolic_derivatives(self, sig):
        """
        d'(tanh(x)) = sech²(x)

        :param sig: x
        :return: Result of sech²(x)
        """
        return np.square(2 / (np.exp(sig) + np.exp(-sig)))


    def simple_error(self, output, output_l, count):
        """
        Simple error
        e.g.
            x1 | x2 | Output dataset | Output answer |  Error
            0  | 0  |       0        |     0.406     | -0.406
            0  | 1  |       1        |     0.432     |  0.568
            1  | 0  |       1        |     0.437     |  0.563
            1  | 1  |       0        |     0.458     | -0.458

            abs mean = 0.49

        :param output: Output dataset
        :param output_l: Output answer
        :param count: Counter
        :return: 1. Difference between output dataset and output answer
                 2. Error
        """
        error_output_layer = output - output_l
        mean_err = np.amax(np.mean(np.abs(error_output_layer), axis=0))
        error = '{0:.2f}'.format(mean_err * 100)
        err100 = float(error)
        print('Iteration %d, loss = %.6f' % (count, mean_err) + '  |  Error: ' + error + '%')
        return error_output_layer, err100


    def base_iris(self):
        """
        Datasets about Iris
        :return: 1. Input datasets
                 2. Output datasets
        """
        base = datasets.load_iris()
        input_d = base.data
        outputValues = base.target
        output = np.empty([len(outputValues), 1], dtype=int)
        for k in range(len(outputValues)):
            if outputValues[k] == 2:
                output[k] = -1
            else:
                output[k] = outputValues[k]
        return input_d, output


    def locate_path(self):
        """
        Locate the path desired.
        :return: Path
        """
        dirlist = os.listdir(".")
        file_n = ''
        for i in dirlist:
            file_n = os.path.abspath(i)
        n = file_n.find('src/')
        path = file_n[:n]
        try:
            os.mkdir(path + 'info')
            os.mkdir(path + 'csv')
        except FileExistsError:
            pass
        finally:
            return path


    def file_csv(self, path, file_name, w):
        """
        Creating file .csv

        :param path: Files path
        :param file_name: File name
        :param w: Data to be written
        :return: None
        """

        path_name = path + 'csv/' + file_name + '.csv'
        try:
            os.remove(path_name)
        except FileNotFoundError:
            pass
        with open(path_name, 'a') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',')
            for linha in w:
                spamwriter.writerow(linha)
            csvfile.flush()
            csvfile.close()

    def info(self):
        """
        Create about information neural network.

        :return: None
        """
        p = self.locate_path()
        base = datasets.load_iris()
        input_data, out_put = self.base_iris()

        df = pd.DataFrame(input_data)
        df.columns = base.feature_names
        df['Class'] = out_put
        df['Answer'] = self.synapse[self.opl]
        df.to_csv(p + 'csv/answer.csv')

        fileInfo = 'info/NeuralNetworkInformation'
        file = open(p + fileInfo, 'wt')
        file.write('************************ Background information ***************************\n\n')
        file.write('Neuron(s) input layer.: %d\n' % len(input_data.T))
        file.write('Neuron(s) first hidden layer.: %d\n' % self.neuronHiddenLayer0)
        file.write('Neuron(s) second hidden layer.: %d\n' % self.neuronHiddenLayer1)
        file.write('Neuron(s) third hidden layer.: %d\n' % self.neuronHiddenLayer2)
        file.write('Neuron(s) output layer.: %d\n' % len(out_put.T))
        file.write('Training time.: %d\n' % self.training_time)
        file.write('Learning rate.: %.6f\n' % self.learning_rate)
        file.write('Momentum.: %d\n' % self.momentum)
        file.write('Error.: ' + str(self.err) + '%\n')
        file.write("""Attribute Information:
           	1. sepal length in cm 
           	2. sepal width in cm 
           	3. petal length in cm 
           	4. petal width in cm 
           	5. Class.:
           		Iris Setosa --> [0]
           		Iris Versicolour --> [1]
           		Iris Virginica --> [-1]
           	""")
        file.write('\n***************************************************************************\n')
        file.write('*************************** Neural Network Answer *************************\n\n')
        file.write(p + 'csv/answer.csv')
        file.write('\n\n***************************************************************************\n')
        file.write('***************************************************************************\n\n')

        file.flush()
        file.close()

        print('\nCreated files!!!')


    def fit(self):
        """
        Trains the neural network with iris dataset.
        :return: None
        """
        # Datastes input and output
        inputData, output = self.base_iris()

        # Random weights
        dctW = {0: 2 * np.random.random((len(inputData.T), self.neuronHiddenLayer0)) - 1,
                1: 2 * np.random.random((self.neuronHiddenLayer0, self.neuronHiddenLayer1)) - 1,
                2: 2 * np.random.random((self.neuronHiddenLayer1, self.neuronHiddenLayer2)) - 1,
                3: 2 * np.random.random((self.neuronHiddenLayer2, len(output.T))) - 1}

        # Bias
        bias = np.array([1])
        dctB = {0: 2 * np.random.random((len(bias), self.neuronHiddenLayer0)) - 1,
                1: 2 * np.random.random((len(bias), self.neuronHiddenLayer1)) - 1,
                2: 2 * np.random.random((len(bias), self.neuronHiddenLayer2)) - 1,
                3: 2 * np.random.random((len(bias), len(output.T))) - 1}

        # Numbers of neuron
        neuronDct = {0: self.neuronHiddenLayer0,
                     1: self.neuronHiddenLayer1,
                     2: self.neuronHiddenLayer2,
                     3: len(output.T)}

        self.synapse = {0: inputData}
        count = 0
        trainingTime = self.training_time

        while trainingTime > 0 and self.err >= self.Err:
            # Layers synapses calculation
            dctIL = {0: inputData}
            for k in range(neuronDct.__len__()):
                biasXWeight = np.empty([len(dctIL[k]), neuronDct[k]], dtype=float)
                aux = k + 1
                dctIL.setdefault(aux, biasXWeight)
                biasA = np.array(np.dot(bias, dctB[k]))
                for i in range(len(dctIL[0])):
                    for j in range(neuronDct[k]):
                        biasXWeight[i][j] = biasA[j]
                addSynapse = np.dot(self.synapse[k], dctW[k]) + biasXWeight
                hiddenLayer = self.hyperbolic(addSynapse)
                self.synapse[aux] = hiddenLayer
                self.opl = aux

            # Error and mean absolut error calculation
            errorOutputLayer, self.err = self.simple_error(output, self.synapse[self.opl], count)

            # Calculation of output delta
            derivedOutput = self.hyperbolic_derivatives(self.synapse[self.opl])
            deltaOutput = errorOutputLayer * derivedOutput

            # Calculation of hidden deltas
            dctDelta = {0: deltaOutput}
            auxK = dctW.__len__()
            for k in range(dctW.__len__() - 1):
                auxK -= 1
                deltaOutputXWeight = dctDelta[k].dot(dctW[auxK].T)
                deltaHidden = deltaOutputXWeight * self.hyperbolic_derivatives(self.synapse[auxK])  # hidden deltas
                dctDelta[k + 1] = deltaHidden

            # Weights bias updates
            auxK = dctB.__len__()
            for k in range(dctDelta.__len__()):
                auxK -= 1
                weightsListBias = []
                add = 0
                deltaT = dctDelta[k].T
                for i in range(len(deltaT)):
                    for j in range(len(dctDelta[k])):
                        add += deltaT[i][j]
                    weightsListBias.append(add)
                biasXdelta = np.array(weightsListBias)
                dctB[auxK] = (dctB[auxK] * self.momentum) + (biasXdelta * self.learning_rate)  # Backpropagation

            # Weights update
            auxK = dctW.__len__()
            for k in range(dctDelta.__len__()):
                auxK -= 1
                layerXWeight = self.synapse[auxK].T.dot(dctDelta[k])
                dctW[auxK] = (dctW[auxK] * self.momentum) + (layerXWeight * self.learning_rate)  # Backpropagation

            count += 1
            trainingTime -= 1

        # Creating file
        p = self.locate_path()
        for i in range(dctW.__len__()):
            index = str(i)
            self.file_csv(p, 'weights' + index, dctW[i])
            self.file_csv(p, 'weights_bias' + index, dctB[i])
        self.info()


    def predict(self, input_data):
        """
        Predict three kind of class, therefore Setosa, Versiclour or Virginica.

        :param input_data:[first, second, third, fourth]
        first --> sepal length in cm
        second --> sepal width in cm
        third --> petal length in cm
        fourth --> petal width in cm

        :return: Type iris class
                [0] --> Iris Setosa
                [1] -- > Iris Versicolour
                [-1] --> Iris Virginica
        """
        p = self.locate_path()

        # Weights
        dctW = {0: np.loadtxt(os.path.join(p + 'csv/weights0.csv'), delimiter=','),
                1: np.loadtxt(os.path.join(p + 'csv/weights1.csv'), delimiter=','),
                2: np.loadtxt(os.path.join(p + 'csv/weights2.csv'), delimiter=','),
                3: np.loadtxt(os.path.join(p + 'csv/weights3.csv'), delimiter=',')}

        # Bias
        bias = np.array([1])
        dctB = {0: np.array([np.loadtxt(os.path.join(p + 'csv/weights_bias0.csv'), delimiter=',')]),
                1: np.array([np.loadtxt(os.path.join(p + 'csv/weights_bias1.csv'), delimiter=',')]),
                2: np.array([np.loadtxt(os.path.join(p + 'csv/weights_bias2.csv'), delimiter=',')]),
                3: np.array([np.loadtxt(os.path.join(p + 'csv/weights_bias3.csv'), delimiter=',')])}

        layers = {0: np.array(input_data)}
        index_output = 0
        for k in range(dctW.__len__()):
            biasA = np.array(np.dot(bias, dctB[k]))
            addSynapse = np.dot(layers[k], dctW[k]) + biasA
            hiddenLayer = self.hyperbolic(addSynapse)
            layers[k + 1] = hiddenLayer
            index_output += 1

        answer = int(np.round(layers[index_output], 1))

        if answer == 0:
            return 'Iris Setosa '
        elif answer == 1:
            return 'Iris Versicolour '
        elif answer == -1:
            return 'Iris Virginica'