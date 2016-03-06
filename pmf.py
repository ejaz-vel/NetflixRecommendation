from scipy.sparse import csr_matrix, coo_matrix, rand, linalg
import math

def computeLoss(ratingDiff, U, V, lambdaU, lambdaV):
	loss = (0.5 * ratingDiff.sum())
	loss += (0.5 * lambdaU * math.pow(linalg.norm(U),2))
	loss += (0.5 * lambdaV * math.pow(linalg.norm(V),2))
	return loss

def computePredictionError(U, V, actualRatings, numUsers, numMovies):
	data = []
	row = []
	col = []
	for userID, movieID, rating in zip(actualRatings.row, actualRatings.col, actualRatings.data):
		user = U.getcol(userID)
		movie = V.getcol(movieID)
		predictedRating = (user.transpose() * movie).data[0]
		row.append(userID)
		col.append(movieID)
		data.append(predictedRating - rating)
	ratingDiff = csr_matrix((data, (row, col)), shape=(numUsers, numMovies))
	return ratingDiff

def randomInitSparse(dim1, dim2):
	matrix = rand(dim1, dim2, density=1, format='csr') - rand(dim1, dim2, density=1, format='csr')
	return matrix

def factorizeMatix(userVectors):
	latentFactors = 20
	numUsers, numMovies = userVectors.shape
	U = randomInitSparse(latentFactors, numUsers)
	V = randomInitSparse(latentFactors, numMovies)
	actualRatings = coo_matrix(userVectors)
	
	lambdaU = 0.1
	lambdaV = 0.1
	learningRate = 0.5
	loss = 99999999
	prevloss = loss
	iteration = 0
	
	while (prevloss - loss) > 0.1 or iteration is 0:
		prevloss = loss
		iteration += 1
		
		print "Computing Prediction Error"
		ratingDiff = computePredictionError(U, V, actualRatings, numUsers, numMovies)
		
		print "Computing gradU and gradV"
		gradientU = (ratingDiff * V.transpose()).transpose() + (lambdaU * U)
		gradientV = (U * ratingDiff) + (lambdaV * V)
		
		print "Computing U and V"
		U = U - (learningRate * gradientU)
		V = V - (learningRate * gradientV)
		
		loss = computeLoss(ratingDiff, U, V, lambdaU, lambdaV)
		print "Loss in Iteration " + str(iteration) + ": " + str(loss)
	return U, V