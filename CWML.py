#Issac Zachariah B922504
from math import exp
from random import seed
from random import random
import numpy as np
import xlrd
import matplotlib.pyplot as plt



path = "CWDataStudent.xls"

#CLASS: Node, used to represent the hidden and output nodes (not the layers, just the individual nodes)
class Node:

    #Initialise the node class with randomised weights and result/data values.    
    def __init__(self, inputs):
        self.weight = [] #list of floats
        #setting all weights to a random value between -1 and 1
        for i in range(inputs + 1):
            self.weight.append((random()*2)-1)
        self.result = 0.0 #stores the latest result of a forward pass
        self.delta = 0.0 #stores the delta values to update its weight
        self.mDelta = [] #The delta momentum
        for i in range(inputs + 1):
            self.mDelta.append(0.0)
        

    #Calculate the activation of the inputs and transfer through a sigmoid function(list of floats)
    def activate(self, data):
        #Adding the bias
        output = self.weight[-1]
        #Calculating data * weight
        for i in range(len(self.weight)-1):
            output += data[i] * self.weight[i]
        return 1.0 / (1.0 + exp(-output))
    
    #A pass through the node (list of floats)
    def propagate(self, data):
        self.result  = self.activate(data)
        #print(result)
        return self.result

        
    #Update the node's weights
    def update(self, step, inputs, momentum):
        for j in range(len(inputs)):
            newWeight = self.weight[j] + (step * self.delta * inputs[j]) + (momentum * self.mDelta[j])
            self.mDelta[j] = newWeight - self.weight[j]  #calculating the weight change for momentum
            self.weight[j] = newWeight
        #Adjusting bias
        newBias = self.weight[-1] + (step * self.delta) + (momentum * self.mDelta[-1])
        self.mDelta[-1] = newBias - self.weight[-1] #calculating the weight change for momentum
        self.weight[-1] = newBias
         
        
    
#CLASS: Machine Learning Peceptron algorithm
class NeuralNetwork:
    #Initialise the Network with the Network (with the layers and nodes initialised too)
    def __init__(self, noLayer, noHiddenNode, inputs, outputs, momentum):

        self.momentum = momentum
        #sum errors used for display
        self.sumError = []
        self.totalRMSE = []
        #Initialise the Network (list of layers)
        self.Network = list()
        #Adding the hidden layers to the network
        for i in range(noLayer):
            tempLayer = list()
            #initialising the nodes in the hidden layers
            for j in range(noHiddenNode[i]):
                #Finds how many inputs the node will have to take
                if (i == 0):
                    noPrev = inputs
                else:
                    noPrev = noHiddenNode[i-1]
                tempLayer.append(Node(noPrev))
            self.Network.append(tempLayer)

        #Add the output layer to the network
        outputLayer = list() #integer
        for i in range(outputs):
            outputLayer.append(Node(noHiddenNode[-1]))
        self.Network.append(outputLayer)
        

    #Forward propagate the inputs to an output (list of floats)
    def forward_pass(self, data):
        #Variable to save results of the nodes from the previous layer to use as an input for the current layer
        inputs = data
        #Going through each layer (forwards)
        for Layer in self.Network:
            tempinputs = [] #Stores the result of each neuron for the next layer to use as inputs
            for Neuron in Layer:
                tempinputs.append(Neuron.propagate(inputs))
                #print(tempinputs)
            inputs = tempinputs
            #print(inputs)
        #print(inputs)
        return inputs

    #Find the derivative of the neuron outputs
    def derivative(self, output):
        return output * (1.0 - output)

    #Backpropagation
    def backward_pass(self, expected):
        #Starting from the output layer and working backwards through the layers
        delta = 0.0 #saves the delta value from the output layer
        for i in reversed(range(len(self.Network))):
            Layer = self.Network[i]
            errors = list()
            #Calculating the error of each neuron
            #For the output layer:
            if i == len(self.Network)-1:
                for j in range(len(Layer)):
                    Neuron = Layer[j]
                    errors.append(expected[j] - Neuron.result)
                #print(errors)
            #For the hidden layer
            else:
                for j in range(len(Layer)):
                    error = 0.0
                    for Neuron in self.Network[i+1]:
                        error += Neuron.weight[j] * delta
                    errors.append(error)
                #print(errors)
            #Finding the delta value for each neuron
            for j in range(len(Layer)):
                Neuron = Layer[j]
                Neuron.delta = errors[j] * self.derivative(Neuron.result)
                #print(Neuron.result)
                #print(Neuron.delta)
                delta += Neuron.delta

    #Update the neuron weights
    def update(self, data, step):
        for i in range(len(self.Network)):
            inputs = data[:-1]
            #If not the first hidden layer
            if i != 0:
                for Neuron in self.Network[i-1]:
                    inputs = [Neuron.result]
            #Adjusting weights weight = weight + step*delta*input + momentum*weight change
            for Neuron in self.Network[i]:
                Neuron.update(step, inputs, self.momentum)

    #Train Network
    def train(self, dataset, startStep, endStep, epochs, allowedError):
        #For each epoch
        for cycle in range(epochs):
            errorSum = 0.0 #errror for this epoch

            #Simulated annealing
            step = endStep + ((startStep - endStep) * (1- (1/(1+ exp(10- (20*(cycle+1))/epochs) ))))            
            for datarow in dataset: #for each row in the dataset
                #Forward pass
                results = self.forward_pass(datarow)
                #print(results)

                #Backward pass
                #Getting the expected values
                expected = []
                for i in range(outputs):
                    expected.append(datarow[-1])
                #print(expected)
                #Calculating the error of the forward pass
                for i in range(len(expected)):
                    errorSum += (expected[i] - results[i])**2

                self.backward_pass(expected)
                self.update(datarow, step)
            
            RMSE = (errorSum/ len(dataset)) ** 0.5
            print('>epoch:%d, step:%.3f, error:%.3f, RMSE:%.3f ' % (cycle, step, errorSum, RMSE))
            self.totalRMSE.append(RMSE)
            self.sumError.append(errorSum)

            #breaks the network if error goes below a certain value to avoid over training
            if errorSum < allowedError:
                break

    #Predict the results, by putting the data through a forward pass.
    def predict(self, dataset):
        errorSum = 0.0
        results = []
        for datarow in dataset:
            result = self.forward_pass(datarow)
            results.append(result)
            
            #Getting the expected values
            expected = []
            for i in range(outputs):
                expected.append(datarow[-1])
            
            #Calculating the error of the forward pass
            for i in range(len(expected)):
                errorSum += (expected[i] - result[i])**2
                
        RMSE = (errorSum/ len(dataset)) ** 0.5
        print("RMSE: %.3f" %(RMSE))
        return results

    #Display the acquired weights (mostly used for testing or debugging)
    def displayWeights(self):
        #Display acquired weights
        for Layer in self.Network:
            print("new layer")
            for Neuron in Layer:
                print(Neuron.weight)


#CLASS: process all the dataset
class processData():

    #Initialise the path, the minMax (for the training data), the columns and rows for the original dataset, get data from excel sheet, split the data for use, and standardise.
    def __init__(self, columns, path):
        self.path = path
        self.minMax = [] #contains min and max values for each column
        self.noColumn = columns
        self.noRows = 0
        
        self.originalDataset = self.getData()
        print("Dataset retrieved")
        self.totalData = self.splitData() #Contains trainData, testData
        print("Dataset split")
        self.standardDataset = self.standardise(self.totalData)
        print("Dataset standardised")        



    #Get the min/max of each column of data
    def getMinMax(self, dataset):
        for i in range(self.noColumn):
            temp = []
            for j in range(len(dataset)):
                #print(i, j)
                temp.append(dataset[j][i])
            self.minMax.append([min(temp), max(temp)])

    #Standardise the dataset
    def standardise(self, dataset):
        sDataset = dataset
        self.getMinMax(sDataset[0])

        #Standardising the data
        for k in range(len(self.totalData)):
            for i in range(self.noColumn):
                tempMin = self.minMax[i][0]
                tempMax = self.minMax[i][1]
                for j in range(len(sDataset[k])):
                    #standardising the dataset between 0.1 and 0.9
                    sDataset[k][j][i] = (0.8* ((self.totalData[k][j][i] - tempMin)/(tempMax - tempMin))) + 0.1
        return sDataset

    #Fetch data from excel sheet
    def getData(self):
        dataset = []
        wb = xlrd.open_workbook(self.path)
        sheet = wb.sheet_by_index(0)

        #Getting each value from the sheet
        for i in range(1,sheet.nrows):
            tempArray = []
            if sheet.cell_value(i,0) != '':
                for j in range(self.noColumn):
                    tempArray.append(sheet.cell_value(i,j))
                dataset.append(tempArray)
        self.noRows = len(dataset)
        return dataset


    #Split the data into 80 20
    def splitData(self):
        totalData = []
        trainData = self.originalDataset
        testData = []
        testSize = round(0.2 * len(self.originalDataset))

        #Do until testData size is accurate
        while True:
            for i in range(len(trainData)):
                if random() > 0.8:
                    if testSize != len(testData):
                        testData.append(trainData[i])
                        trainData.pop(i)
                        break
            if testSize == len(testData):
                break

        totalData.append(trainData)
        totalData.append(testData)
        return totalData
    
    #Destandardise the results
    def destandardise(self, inputs):
        values = []

        #destandardising the data
        tempMin = self.minMax[-1][0]
        tempMax = self.minMax[-1][1]
        for i in range(len(inputs)):
            values.append( (((inputs[i][0] - 0.1)/0.8) * (tempMax - tempMin)) + tempMin)
        return values


#Seed the random values
seed(1)

#Initialise the dataset given the columns
mainData = processData(9,path)

#for i in range(len(mainData.standardDataset)):
#    print(mainData.standardDataset[i][-1])

#variables
startStep = 0.9
endStep = 0.01
momentum = 0.9
epochs = 5000
noLayer = 1
noHiddenNode = [8]

outputs = 1
inputs = mainData.noColumn - outputs
allowedError = 0.3


#initialise the network
net = NeuralNetwork(noLayer, noHiddenNode, inputs, outputs, momentum)
net.train(mainData.totalData[0], startStep, endStep, epochs, allowedError)


net.displayWeights()

result = []
actual = []
#print(mainData.totalData[1])

prediction = net.predict(mainData.totalData[1])

#print(result)
DSresult = mainData.destandardise(prediction)
print(DSresult)
actual = mainData.destandardise(mainData.totalData[1])
print(actual)


plt.plot(net.totalRMSE)
#plt.plot(net.sumError)
plt.show()


print("End")






