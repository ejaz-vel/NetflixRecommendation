from scipy.sparse import csr_matrix
import math
import numpy as np

# Read the Latent Factor Matrix from the file
# Depending on what file is passed as input, this API will load either the user or movie latent factors matrix
# @fileName: The file which stores the Latent Factors of the entity (user/movie)
# @numLatentFactors: The number of Latent Factors used while generating the above file
# @numEntities: This refers to either the number of users or number of movies
def readLatentFactors(fileName, numLatentFactors, numEntities):
	X = np.empty([numEntities,numLatentFactors])
	f = open(fileName)
	num = 0
	for line in iter(f):
		featureID = 0
		factors = line.split()
		for factor in factors:
			X[num][featureID] = float(factor.strip())
			featureID += 1
		num += 1
	f.close()
	return X

# Given the Latent Factors for the User and Movie, generate the feature Vector
# @U: The User latent factor matrix
# @V: The movie latent factor matrix
# @user: The userID for which the feature needs to be generated
# @movie: The movieID for which the feature needs to be generated
def getFeatureVector(U, V, user, movie):
	userVector = U.T[user]
	movieVector = V.T[movie]
	# Perform element-wise multiplication, to generate the feature vector
	return np.multiply(userVector, movieVector)

# This API is mostly used to extract the weights learned by the SVM module in liblinear
# @modelFile: The model learned by the SVM during training
# @numLatentFactors: Number of latent factors used to train the SVM
def getWeightsFromModelFile(modelFile, numLatentFactors):
	weights = np.random.rand(numLatentFactors)
	f = open(modelFile)
	readWeights = False
	index = 0
	for line in iter(f):
		# The model file starts listing the weights after this keyword
		if line.strip() == "w":
			readWeights = True
			continue
			
		if readWeights is True:
			weights[index] = float(line.strip())
			index += 1
	f.close()
	return weights

# Given the Latent Factors for the User and Movie, generate the Training Data
# @U: The User latent factor matrix
# @V: The movie latent factor matrix
def generateLetorFeatures(U, V):
	row = []
	col = []
	data = []
	
	numUsers = 0
	numMovies = 0
	# The training file containing the observed ratings
	f = open("HW4_data/train.csv")
	
	for line in iter(f):
		element = line.split(",")
		movieID = int(element[0])
		userID = int(element[1])
		rating = int(element[2])
		
		row.append(userID)
		col.append(movieID)
		data.append(rating)
		
		numUsers = max(userID, numUsers)
		numMovies = max(movieID, numMovies)
	f.close()
	numUsers += 1
	numMovies += 1
	# Load the observed ratings matrix into memory
	userVectors = csr_matrix((data, (row, col)), shape=(numUsers, numMovies))
	
	# The features will be written to this file
	featureFile = open("features.txt", "w")
	for userID in range(numUsers):
		print "Examining user: " + str(userID)
		user = userVectors.getrow(userID)
		for i in range(len(user.data)):
			for j in range(len(user.data)):
				ri = user.data[i]
				rj = user.data[j]
				# Only generate features when rating difference is 4.
				if abs(ri - rj) == 4:
					xi = getFeatureVector(U, V, userID, user.indices[i])
					xj = getFeatureVector(U, V, userID, user.indices[j])
					feat = xi - xj
					y = math.copysign(1, ri - rj)
					line = str(y) + " "
					featureID = 1
					for num in feat:
						line += str(featureID) + ":"
						line += str(num) + " "
						featureID += 1
					featureFile.write(line.strip() + "\n")
	featureFile.close()

# Generate the Letor features for the test data.
# These features are used by the LR-Letor module to make predictions
# @inputFilePath: The file containing the user, movie pairs for which the rating needs to be predicted
# @numLatentFactors: Number of latent factors to be used to generate the features
# @numUsers: Total number of users in our collection
# @numMovies: Total number of moview in our collection
def generateTestFeatures(inputFilePath, numLatentFactors, numUsers, numMovies):
	# ASSUMPTION: The user and movie latent factors have already been generated
	U = readLatentFactors("userFactors.txt", numLatentFactors, numUsers)
	V = readLatentFactors("movieFactors.txt", numLatentFactors, numMovies)
	
	testFeatureFile = open("testFeatures.txt", "w")
	f = open(inputFilePath)
	for line in iter(f):
		movieID = int(line.split(",")[0])
		userID = int(line.split(",")[1])
		feat = getFeatureVector(U.T, V.T, userID, movieID)
		l = "0 "
		featureID = 1
		for num in feat:
			l += str(featureID) + ":"
			l += str(num) + " "
			featureID += 1
		testFeatureFile.write(l.strip() + "\n")
	f.close()
	testFeatureFile.close()

# Directly Make predictions from the user-movie latent factors, without using the features
# This will help us compare the difference between the RMSE and NDCG values.
# @inputFilePath: The file containing the user, movie pairs for which the rating needs to be predicted
# @numLatentFactors: Number of latent factors used the to generate the user and movie latent factors
# @numUsers: Total number of users in our collection
# @numMovies: Total number of moview in our collection
def makeRatingPredictions(inputFilePath, numLatentFactors, numUsers, numMovies):
	U = readLatentFactors("userFactors.txt", numLatentFactors, numUsers)
	V = readLatentFactors("movieFactors.txt", numLatentFactors, numMovies)
	
	# Multiply the 2 matrices to generate the ratings for each user, movie combination
	prediction = np.dot(U, V.transpose())
	
	ratingsFile = open("ratings.txt", "w")
	f = open(inputFilePath)
	for line in iter(f):
		movieID = int(line.split(",")[0])
		userID = int(line.split(",")[1])
		rating = prediction[userID, movieID]
		rating += 3
		ratingsFile.write(str(rating) + "\n")
	f.close()
	ratingsFile.close()
	
# Generate the ranking scores for the user-movie pairs using feature weights found during the training phase
# To generate the ranking we need to generate the feature vector for each user-movie pair. Used for the Rank-SVM module
# @inputFilePath: The file containing the user, movie pairs for which the rating needs to be predicted
# @weights: The weights learned from the training phase
# @numLatentFactors: Number of latent factors used the to generate the user and movie latent factors
# @numUsers: Total number of users in our collection
# @numMovies: Total number of moview in our collection
def makeRankingPredictions(inputFilePath, weights, numLatentFactors, numUsers, numMovies):
	U = readLatentFactors("userFactors.txt", numLatentFactors, numUsers)
	V = readLatentFactors("movieFactors.txt", numLatentFactors, numMovies)
	
	rankingFile = open("predictions.txt", "w")
	f = open(inputFilePath)
	for line in iter(f):
		movieID = int(line.split(",")[0])
		userID = int(line.split(",")[1])
		feat = getFeatureVector(U.T, V.T, userID, movieID)
		prediction = np.dot(weights, feat)
		rankingFile.write(str(prediction) + "\n")
	f.close()
	rankingFile.close()