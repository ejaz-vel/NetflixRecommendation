from scipy.sparse import csr_matrix
import math
import time
import os
import pmf

imputationConstant = 3
globalUserDict = {}
globalMovieDict = {}

def computeCosineSimilarity(vector1, vector2):
	sim = vector1.dot(vector2.T)
	if sim.nnz == 0:
		return 0.0
	else:
		norm1 = math.sqrt(vector1.multiply(vector1).sum())
		norm2 = math.sqrt(vector2.multiply(vector2).sum())
		return sim.data[0]/(norm1 * norm2)

def computeDotProduct(vector1, vector2):
	sim = vector1.dot(vector2.T)
	if sim.nnz == 0:
		return 0.0
	else:
		return float(sim.data[0])

def standardizeMatrixRow(data, lastUpdated, current):
	mean = (sum(data[lastUpdated:current]) + 0.0) / len(data[lastUpdated:current])
	valuesRange = math.sqrt(sum(map(lambda x:x*x,data[lastUpdated:current])))
	data[lastUpdated:current] = [(x - mean)/valuesRange for x in data[lastUpdated:current]]

def getUserVectors(standardizationRequired = False):
	row = []
	col = []
	data = []
	
	numRows = 0
	numMovies = 0
	lastUpdatedIndex = 0
	currentIndex = 0
	f = open("HW4_data/dev.queries")
	for line in iter(f):
		userData = line.split()
		userID = int(userData[0])
		lastUpdatedIndex = currentIndex
		for rating in userData[1:]:
			movieID = int(rating.split(":")[0])
			rating = int(rating.split(":")[1]) - imputationConstant
			if(rating == 0):
				continue
			if numMovies < movieID:
				numMovies = movieID
			row.append(userID)
			col.append(movieID)
			data.append(rating)
			currentIndex += 1
		if standardizationRequired is True and lastUpdatedIndex != currentIndex:
			standardizeMatrixRow(data, lastUpdatedIndex, currentIndex)
		if numRows < userID:
			numRows = userID
	f.close()
		
	numRows += 1
	numMovies += 1
	userVectors = csr_matrix((data, (row, col)), shape=(numRows, numMovies))
	
	return numRows, numMovies, userVectors
		
def computeItemAssociationsUsingDotProduct(userVectors):
	itemAssociations = userVectors.T * userVectors
	return itemAssociations

def computeItemAssociationsUsingCosineSimilarity(userVectors, numMovies):
	row = []
	col = []
	data = []
	
	if os.path.exists("itemAssociations.txt"):
		f = open("itemAssociations.txt")
		for line in iter(f):
			element = line.split()
			if float(element[2]) > 0:
				row.append(int(element[0]))
				col.append(int(element[1]))
				data.append(float(element[2]))
	else:
		for movieIDRow in range(numMovies):
			if userVectors.getcol(movieIDRow).nnz == 0:
				continue
			print "Filling Row: " + str(movieIDRow)
			for movieIDCol in range(numMovies):
				if userVectors.getcol(movieIDCol).nnz == 0:
					continue
				row.append(movieIDRow)
				col.append(movieIDCol)
				data.append(computeCosineSimilarity(userVectors.getcol(movieIDRow).T, userVectors.getcol(movieIDCol).T))
			
		associationsFile = open("itemAssociations.txt", "w")
		for index in range(len(row)):
			associationsFile.write(str(row[index]) + " " + str(col[index]) + " " + str(data[index]) + "\n")
		associationsFile.close()
		
	itemAssociations = csr_matrix((data, (row, col)), shape=(numMovies, numMovies))
	return itemAssociations

def getTopK(neighbourList, k):
	kList = []
	for i in range(k):
		if len(neighbourList) == 0:
			break

		pos = neighbourList.index(max(neighbourList, key=lambda x: x[1]))
		kList.append(neighbourList[pos])
		neighbourList.pop(pos)
	return kList

def findKSimilarUsers(k, userID, userVectors, numUsers, useDotProduct):
	if not globalUserDict.has_key(userID):
		neighbourList = []
		currentUser = userVectors.getrow(userID)
		for i in range(numUsers):
			sampleUser = userVectors.getrow(i)
			if i == userID or sampleUser.nnz == 0:
				continue
			
			if useDotProduct is True:
				similarity = computeDotProduct(currentUser, sampleUser)
			else:
				similarity = computeCosineSimilarity(currentUser, sampleUser)
			neighbourList.append((i, similarity))
		globalUserDict[userID] = getTopK(neighbourList, k)
	return globalUserDict[userID]

def findKSimilarMovies(k, movieID, itemAssociations, numMovies):
	if not globalMovieDict.has_key(movieID):
		neighbourList = []
		currentMovie = itemAssociations.getrow(movieID)
		for i in range(numMovies):
			if i == movieID:
				continue
			
			if currentMovie.getcol(i).nnz != 0:
				neighbourList.append((i, currentMovie.getcol(i).data[0]))
		globalMovieDict[movieID] = getTopK(neighbourList, k)
	return globalMovieDict[movieID]

def computePredictionUsingMean(movieID, neighbourList, userVectors):
	sumOfRatings = 0
	k = len(neighbourList)
	
	if len(neighbourList) != 0:
		weight = 1.0/len(neighbourList)
		
	for neighbour in neighbourList:
		user = userVectors.getrow(neighbour[0])
		movie = user.getcol(movieID)
		if movie.nnz != 0:
			sumOfRatings += (weight * movie.data[0])
	print sumOfRatings
	return sumOfRatings + imputationConstant

def computePredictionUsingWeightedMean(movieID, neighbourList, userVectors):
	sumOfRatings = 0
	runningSum = 0
	
	for neighbour in neighbourList:
		similarity = neighbour[1]
		runningSum += similarity
		user = userVectors.getrow(neighbour[0])
		movie = user.getcol(movieID)
		
		if movie.nnz != 0:
			sumOfRatings += (similarity * movie.data[0])
	
	if runningSum != 0:
		sumOfRatings /= runningSum
	print sumOfRatings
	return sumOfRatings + imputationConstant

def predictRatingsUsingUserSimilarity(useDotProduct=True, useWeightedMean=True):
	[numUsers, numMovies, userVectors] = getUserVectors()
	ratingsFile = open("ratings.txt", "w")
	f = open("HW4_data/dev.csv")
	
	beforeTime = time.time()
	for line in iter(f):
		movieID = int(line.split(",")[0])
		userID = int(line.split(",")[1])
		print "Computing Prediction for user-movie: " + str(userID) + "-" + str(movieID)
		neighbourList = findKSimilarUsers(100, userID, userVectors, numUsers, useDotProduct)
		
		rating = 0
		if useWeightedMean is True:
			rating = computePredictionUsingWeightedMean(movieID, neighbourList, userVectors)
		else:
			rating = computePredictionUsingMean(movieID, neighbourList, userVectors)
		ratingsFile.write(str(rating) + "\n")
	f.close()
	ratingsFile.close()
	afterTime = time.time()
	print "Time Taken for online Computation: " + str(afterTime - beforeTime)

def getMeanRating(userID, movieID, neighbourList, itemAssociations, userVectors):
	sumOfRatings = 0
	runningSum = 0
	user = userVectors.getrow(userID)
	
	if len(neighbourList) != 0:
		weight = 1.0/len(neighbourList)

	for neighbour in neighbourList:
		movie = user.getcol(neighbour[0])
		if movie.nnz != 0:
			sumOfRatings += (weight * movie.data[0])
	print sumOfRatings
	return sumOfRatings + imputationConstant
	
def getWeighedMeanRating(userID, movieID, neighbourList, itemAssociations, userVectors):
	sumOfRatings = 0
	runningSum = 0
	user = userVectors.getrow(userID)
	
	for neighbour in neighbourList:
		movie = user.getcol(neighbour[0])
		weight = itemAssociations.getrow(movieID).getcol(neighbour[0])
		
		if weight.nnz != 0:
			runningSum += weight.data[0]
			if movie.nnz != 0:
				sumOfRatings += (weight.data[0] * movie.data[0])
				
	if runningSum != 0:
		sumOfRatings /= (runningSum + 0.0)
	print sumOfRatings
	return sumOfRatings + imputationConstant

def predictRatingsUsingMovieSimilarity(useDotProduct=True, useWeightedMean=True, standardizationRequired=False):
	[numUsers, numMovies, userVectors] = getUserVectors(standardizationRequired)
	itemAssociations = []
	if useDotProduct is True:
		itemAssociations = computeItemAssociationsUsingDotProduct(userVectors)
	else:
		itemAssociations = computeItemAssociationsUsingCosineSimilarity(userVectors, numMovies)

	ratingsFile = open("ratings.txt", "w")
	f = open("HW4_data/dev.csv")

	beforeTime = time.time()
	for line in iter(f):
		movieID = int(line.split(",")[0])
		userID = int(line.split(",")[1])
		print "Computing Prediction for user-movie: " + str(userID) + "-" + str(movieID)
		neighbourList = findKSimilarMovies(20, movieID, itemAssociations, numMovies)
		rating = 0
		if useWeightedMean is True:
			rating = getWeighedMeanRating(userID, movieID, neighbourList, itemAssociations, userVectors)
		else:
			rating = getMeanRating(userID, movieID, neighbourList, itemAssociations, userVectors)
		ratingsFile.write(str(rating) + "\n")
	f.close()
	ratingsFile.close()
	afterTime = time.time()
	print "Time Taken for online Computation: " + str(afterTime - beforeTime)

def predictRatingsByPMF():
	[numUsers, numMovies, userVectors] = getUserVectors()
	[U, V] = pmf.factorizeMatix(userVectors)
	prediction = U.transpose() * V
	
	ratingsFile = open("ratings.txt", "w")
	f = open("HW4_data/dev.csv")
	beforeTime = time.time()
	for line in iter(f):
		movieID = int(line.split(",")[0])
		userID = int(line.split(",")[1])
		rating = prediction.getrow(userID).getcol(movieID).data[0] + imputationConstant
		print "Computing Prediction for user-movie: " + str(userID) + "-" + str(movieID) + "=" + str(rating)
		ratingsFile.write(str(rating) + "\n")
	f.close()
	ratingsFile.close()
	afterTime = time.time()
	print "Time Taken for online Computation: " + str(afterTime - beforeTime)
