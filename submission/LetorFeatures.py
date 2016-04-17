from scipy.sparse import csr_matrix
import math
import numpy as np

def getFeatureVector(U, V, user, movie):
	userVector = U.T[user]
	movieVector = V.T[movie]
	return np.multiply(userVector, movieVector)

def generateLetorFeatures(U, V):
	row = []
	col = []
	data = []
	
	numUsers = 0
	numMovies = 0
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
	userVectors = csr_matrix((data, (row, col)), shape=(numUsers, numMovies))
	
	featureFile = open("features.txt", "w")
	for userID in range(numUsers):
		print "Examining user: " + str(userID)
		user = userVectors.getrow(userID)
		for i in range(len(user.data)):
			for j in range(len(user.data)):
				ri = user.data[i]
				rj = user.data[j]
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

def generateTestFeatures(inputFileName, numLatentFactors, numUsers, numMovies):
	U = readLatentFactors("userFactors.txt", numLatentFactors, numUsers)
	V = readLatentFactors("movieFactors.txt", numLatentFactors, numMovies)
	
	testFeatureFile = open("testFeatures.txt", "w")
	f = open("HW4_data/" + inputFileName)
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

def postProcessOutput():
	f = open("output.txt")
	test = open("testpredictions.txt", "w")
	lineNum = 0
	for line in iter(f):
		if lineNum != 0:
			data = line.split()
			test.write(data[1].strip() + "\n")
		lineNum += 1
	f.close()
	test.close()