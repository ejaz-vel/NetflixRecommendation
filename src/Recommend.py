from scipy.sparse import csr_matrix
import math
import time
import os
import pmf
import numpy as np
import LetorFeatures

trainingFile = "data/dev.queries"
testingFile = "data/dev.csv"

imputationConstant = 3
globalUserDict = {}
globalMovieDict = {}
userAverageRating = {}

# API for computing the cosine similarity between two vectors
def computeCosineSimilarity(vector1, vector2):
	sim = vector1.dot(vector2.T)
	if sim.nnz == 0:
		return 0.0
	else:
		norm1 = math.sqrt(vector1.multiply(vector1).sum())
		norm2 = math.sqrt(vector2.multiply(vector2).sum())
		return sim.data[0]/(norm1 * norm2)

# API for computing the dot product similarity between two vectors
def computeDotProduct(vector1, vector2):
	sim = vector1.dot(vector2.T)
	if sim.nnz == 0:
		return 0.0
	else:
		return float(sim.data[0])

# API for standardizing the userVectors based on average user rating
def standardizeMatrixRow(userID, data, lastUpdated, current):
	mean = (sum(data[lastUpdated:current]) + 0.0) / len(data[lastUpdated:current])
	# Store the average user rating in a global dictionary
	userAverageRating[userID] = mean
	data[lastUpdated:current] = [(x - mean) for x in data[lastUpdated:current]]

# API for reading the user-movie matrix from the input data
# standardizationRequired: This parameter controls the behavior of normalizing/standardinzing the the user vectors
def getUserVectors(standardizationRequired = False):
	row = []
	col = []
	data = []
	
	numRows = 0
	numMovies = 0
	lastUpdatedIndex = 0
	currentIndex = 0
	f = open(trainingFile)
	for line in iter(f):
		userData = line.split()
		userID = int(userData[0])
		lastUpdatedIndex = currentIndex
		for rating in userData[1:]:
			movieID = int(rating.split(":")[0])
			rating = int(rating.split(":")[1])
			
			# Perform Imputation
			if not standardizationRequired is True:
				rating -= imputationConstant
			# We only care about non-zero movie ratings.
			if(rating == 0):
				continue
				
			if numMovies < movieID:
				numMovies = movieID
			row.append(userID)
			col.append(movieID)
			data.append(rating)
			currentIndex += 1
		# Standardize the user vectors
		if standardizationRequired is True and lastUpdatedIndex != currentIndex:
			standardizeMatrixRow(userID, data, lastUpdatedIndex, currentIndex)
			
		if numRows < userID:
			numRows = userID
	f.close()
		
	numRows += 1
	numMovies += 1
	userVectors = csr_matrix((data, (row, col)), shape=(numRows, numMovies))
	return numRows, numMovies, userVectors

# API for building the movie-movie similarity model using the dot product
def computeItemAssociationsUsingDotProduct(userVectors):
	itemAssociations = userVectors.T * userVectors
	return itemAssociations

# API for building the movie-movie similarity model using the cosine similarity
def computeItemAssociationsUsingCosineSimilarity(userVectors, numMovies):
	row = []
	col = []
	data = []
	
	# If the model has already been built earlier, then load it into memory.
	# Else, we would explicitly have to build the model.
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
		
		# Store the model in a file on the disk, so that it can be reused later
		associationsFile = open("itemAssociations.txt", "w")
		for index in range(len(row)):
			associationsFile.write(str(row[index]) + " " + str(col[index]) + " " + str(data[index]) + "\n")
		associationsFile.close()
		
	itemAssociations = csr_matrix((data, (row, col)), shape=(numMovies, numMovies))
	return itemAssociations

# Retrieve the top K elements from the neighbourList
def getTopK(neighbourList, k):
	kList = []
	for i in range(k):
		if len(neighbourList) == 0:
			break

		pos = neighbourList.index(max(neighbourList, key=lambda x: x[1]))
		kList.append(neighbourList[pos])
		neighbourList.pop(pos)
	return kList

# Find the top K Similars users for the user provided by the parameter 'userID'
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

# Find the top K Similars movies for the movie provided by the parameter 'movieID'
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

# Use the mean weighting scheme to compute the rating for the movieID.
def computePredictionUsingMean(userID, movieID, neighbourList, userVectors):
	sumOfRatings = 0
	k = len(neighbourList)
	
	if len(neighbourList) != 0:
		weight = 1.0/len(neighbourList)
		
	for neighbour in neighbourList:
		user = userVectors.getrow(neighbour[0])
		movie = user.getcol(movieID)
		if movie.nnz != 0:
			sumOfRatings += (weight * movie.data[0])
			
	if userID in userAverageRating:
		return sumOfRatings + userAverageRating[userID]
	return sumOfRatings + imputationConstant

# Use the weighted mean weighting scheme to compute the rating for the movieID.
def computePredictionUsingWeightedMean(userID, movieID, neighbourList, userVectors):
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
		
	if userID in userAverageRating:
		return sumOfRatings + userAverageRating[userID]
	return sumOfRatings + imputationConstant

# API for user-user similarity: You can use the arguments of this API to decide what similarity metric or weighting scheme to use. 
def predictRatingsUsingUserSimilarity(useDotProduct=True, useWeightedMean=True, standardizationRequired=False):
	[numUsers, numMovies, userVectors] = getUserVectors(standardizationRequired)
	ratingsFile = open("ratings.txt", "w")
	f = open(testingFile)
	
	beforeTime = time.time()
	for line in iter(f):
		movieID = int(line.split(",")[0])
		userID = int(line.split(",")[1])
		print "Computing Prediction for user-movie: " + str(userID) + "-" + str(movieID)
		neighbourList = findKSimilarUsers(10, userID, userVectors, numUsers, useDotProduct)
		
		rating = 0
		if useWeightedMean is True:
			rating = computePredictionUsingWeightedMean(userID, movieID, neighbourList, userVectors)
		else:
			rating = computePredictionUsingMean(userID, movieID, neighbourList, userVectors)
		ratingsFile.write(str(rating) + "\n")
	f.close()
	ratingsFile.close()
	afterTime = time.time()
	print "Time Taken for online Computation: " + str(afterTime - beforeTime)

# Use the mean weighting scheme to compute the rating for the userID - movieID combination.
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
			
	if userID in userAverageRating:
		return sumOfRatings + userAverageRating[userID]
	return sumOfRatings + imputationConstant

# Use the weighted mean weighting scheme to compute the rating for the userID - movieID combination.
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
		
	if userID in userAverageRating:
		return sumOfRatings + userAverageRating[userID]
	return sumOfRatings + imputationConstant

# API for movie-movie similarity: You can use the arguments of this API to decide what similarity metric or weighting scheme to use. 
# Also, you can decide whether to use standardized user vectors or not, using this API.  
def predictRatingsUsingMovieSimilarity(useDotProduct=True, useWeightedMean=True, standardizationRequired=False):
	[numUsers, numMovies, userVectors] = getUserVectors(standardizationRequired)
		
	itemAssociations = []
	if useDotProduct is True:
		itemAssociations = computeItemAssociationsUsingDotProduct(userVectors)
	else:
		itemAssociations = computeItemAssociationsUsingCosineSimilarity(userVectors, numMovies)

	ratingsFile = open("ratingsMovies.txt", "w")
	f = open(testingFile)

	beforeTime = time.time()
	for line in iter(f):
		movieID = int(line.split(",")[0])
		userID = int(line.split(",")[1])
		print "Computing Prediction for user-movie: " + str(userID) + "-" + str(movieID)
		neighbourList = findKSimilarMovies(10, movieID, itemAssociations, numMovies)
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

# API for PMF: You can use the arguments of this API to decide whether to use standardized user vectors or not
def predictRatingsByPMF(standardizationRequired=False):
	[numUsers, numMovies, userVectors] = getUserVectors(standardizationRequired)
	beforeTime = time.time()
	[U, V] = pmf.factorizeMatix(userVectors)
	prediction = np.dot(U.transpose(), V)
	
	ratingsFile = open("ratings.txt", "w")
	f = open(testingFile)
	for line in iter(f):
		movieID = int(line.split(",")[0])
		userID = int(line.split(",")[1])
		rating = prediction[userID, movieID]
		if userID in userAverageRating:
			rating += userAverageRating[userID]
		else:
			rating += imputationConstant
		ratingsFile.write(str(rating) + "\n")
	f.close()
	ratingsFile.close()
	afterTime = time.time()
	print "Time Taken for online Computation: " + str(afterTime - beforeTime)

# Writee the learned user and movie factors to disk
def writeUserAndMovieFactors(U, V):
	[factors, users] = U.shape
	movies = V.shape[1]
	
	userFactors = open("userFactors.txt", "w")
	for userID in range(users):
		line = ""
		for num in U.T[userID]:
			line += str(num) + " "
		userFactors.write(line.strip() + "\n")
	userFactors.close()
	
	movieFactors = open("movieFactors.txt", "w")
	for movieID in range(movies):
		line = ""
		for num in V.T[movieID]:
			line += str(num) + " "
		movieFactors.write(line.strip() + "\n")
	movieFactors.close()

# API for generating the features required for Learning to Rank
def generateFeaturesForLetor(standardizationRequired=False):
	[numUsers, numMovies, userVectors] = getUserVectors(standardizationRequired)
	[U, V] = pmf.factorizeMatix(userVectors)
	writeUserAndMovieFactors(U, V)
	LetorFeatures.generateLetorFeatures(U, V)