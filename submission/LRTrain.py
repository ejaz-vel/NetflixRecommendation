import math
import numpy as np
from scipy.special import expit

# Calculates the Value of the loss function contributed by this prediction
def getLoss(actualLabel, predictedLabel):
	loss = 0
	if predictedLabel > 0 and predictedLabel < 1:
		loss = (actualLabel * math.log(predictedLabel)) + ((1 - actualLabel) * math.log(1 - predictedLabel))
	return loss

# Controls the convergence condition.
# The threshold parameter needs to be carefully tuned.
def isConverged(currentLoss, previousLoss, threshold, iteration):
	if (previousLoss is None) or (iteration < 5):
		return False
		
	lossDiff = currentLoss - previousLoss
	if lossDiff > threshold:
		return False
	return True

# Converts a line from the feature vector file to a feature stored in a numpy object
def constructFeature(numFeatures, data):
	featureVector = np.zeros(numFeatures + 1)
	label = int(float(data[0]))
	features = data[1:]
	
	featureVector[0] = 1.0
	for element in features:
		st = element.split(":")
		featureVector[int(st[0])] = float(st[1])
	return featureVector, label

# Performs Stochastic Gradient Descent.
# The batchSize parameter controls the number of elements in one batch before the weights are updated
def trainModelWithStochasticGD(featureVectorFile, numFeatures, batchSize, dataSize):
	print "Training the model"
	
	w = np.random.rand(numFeatures + 1)
	learningRate = 0.005
	regularizationFactor = 0.001
	previousLoss = None
	globalLoss = -10000
	numIteration = 0
	
	while not isConverged(globalLoss, previousLoss, 0.0001, numIteration):
		previousLoss = globalLoss
		globalLoss = 0
		gradient = np.zeros(numFeatures + 1)
		loss = 0
		numIteration += 1
		
		iteration = 1
		f = open(featureVectorFile)
		for line in iter(f):
			data = line.split()
			x, output = constructFeature(numFeatures, data)
			y = int(output == 1)
			p = expit(np.dot(w, x.transpose()))
			gradient += ((y - p) * x)
			loss += getLoss(y, p) / dataSize
			
			if iteration % batchSize == 0:
				gradient -= regularizationFactor * w
				loss -= regularizationFactor * np.sum(w ** 2) * ((batchSize + 0.0) / dataSize)
				w += learningRate * gradient
				gradient = np.zeros(numFeatures + 1)
				globalLoss += loss
				loss = 0
			iteration += 1
		numUpdates = iteration % batchSize
		gradient -= regularizationFactor * w
		loss -= regularizationFactor * np.sum(w ** 2) * ((numUpdates + 0.0) / dataSize)
		w += learningRate * gradient
		
		f.close()
		globalLoss += loss
		print globalLoss
	return w

# Performs Batch Gradient Descent using the entire dataset.
# Data is NOT loaded into memory. It is continously streamed from the featureVectorFile
def trainModelWithBatchGD(featureVectorFile, numFeatures, dataSize):
	print "Training the model"	
	w = np.random.rand(numFeatures + 1)
	currentLoss = -10000
	previousLoss = None
	learningRate = 1.4
	regularizationFactor = 0.2
	
	iteration = 0
	while not isConverged(currentLoss, previousLoss, 0.00001, iteration):
		f = open(featureVectorFile)
		previousLoss = currentLoss
		currentLoss = 0
		gradient = np.zeros(numFeatures + 1)
		iteration += 1
			
		for line in iter(f):
			data = line.split()
			x, output = constructFeature(numFeatures, data)
			y = int(output == 1)
			p = expit(np.dot(w, x.transpose()))
			gradient += ((y - p) * x) / dataSize
			currentLoss += getLoss(y, p) / dataSize
		gradient -= regularizationFactor * w
		currentLoss -= regularizationFactor * np.sum(w ** 2)
		w += learningRate * gradient
		print "Iteration " + str(iteration) + ": " + str(currentLoss)
		f.close()
	return w

# Takes the learnt weights and the input featureVectorFile to perform predictions.
# We don't care about the binary class predicted. 
# Hence, we don't convert the value of the sigmoid function to the corresponding class.
def classify(weights, numFeatures, featureVectorFile):
	predFile = "prediction.txt"
	f = open(featureVectorFile)
	predictionFile = open(predFile, "w")
	
	for line in iter(f):
		x, output = constructFeature(numFeatures, line.split())
		prediction = expit(np.dot(weights, x.transpose()))
		predictionFile.write(str(prediction) + "\n")
	f.close()
	predictionFile.close()