from scipy.sparse import csr_matrix, coo_matrix, linalg
import math
import numpy as np

def computeLoss(ratingDiff, U, V, lambdaU, lambdaV):
	loss = (0.5 * math.pow(linalg.norm(ratingDiff),2))
	loss += (0.5 * lambdaU * math.pow(np.linalg.norm(U),2))
	loss += (0.5 * lambdaV * math.pow(np.linalg.norm(V),2))
	return loss/(ratingDiff.nnz + 0.0)
                                                              
def performGD(U, V, actualRatings, lambdaU, lambdaV):
	userGradient = {}
	movieGradient = {}
	row = []
	col = []
	data = []
	
	for userID, movieID, rating in zip(actualRatings.row, actualRatings.col, actualRatings.data):
		user = U[:,userID]
		movie = V[:,movieID]
		predictedRating = user.dot(movie)
		predictionError = predictedRating - rating
		row.append(userID)
		col.append(movieID)
		data.append(predictionError)
		
		if userID in userGradient:
			userGradient[userID] += (predictionError * movie)
		else:
			userGradient[userID] = (predictionError * movie) + (lambdaU * user)
			
		if movieID in movieGradient:
			movieGradient[movieID] += (predictionError * user)
		else:
			movieGradient[movieID] = (predictionError * user) + (lambdaV * movie)
	ratingDiff = csr_matrix((data, (row, col)), shape=actualRatings.shape)
	return userGradient, movieGradient, ratingDiff

def factorizeMatix(userVectors):
	latentFactors = 30
	[numUsers, numMovies] = userVectors.shape
	U = np.random.rand(latentFactors, numUsers)
	V = np.random.rand(latentFactors, numMovies)
	actualRatings = coo_matrix(userVectors)
	
	lambdaU = 0.1
	lambdaV = 0.1
	learningRate = 0.00018
	loss = 99
	prevloss = loss
	iteration = 0
	
	while (prevloss - loss) > 0.0001 or iteration is 0:
		prevloss = loss
		iteration += 1
		[userGradient, movieGradient, ratingDiff] = performGD(U, V, actualRatings, lambdaU, lambdaV)
		
		for userID in range(numUsers):
			if userID in userGradient:
				U[:,userID] = U[:,userID] - (learningRate * userGradient[userID])
		for movieID in range(numMovies):
			if movieID in movieGradient:
				V[:,movieID] = V[:,movieID] - (learningRate * movieGradient[movieID])
		
		loss = computeLoss(ratingDiff, U, V, lambdaU, lambdaV)
		print "Loss in Iteration " + str(iteration) + ": " + str(loss)
	return U, V