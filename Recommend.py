from scipy.sparse import csr_matrix
import math

def normalizeMatrixRow(data, lastUpdated, currentIndex, sumOfSquares):
	normFactor = math.sqrt(sumOfSquares)
	data[lastUpdated:currentIndex] = [x/normFactor for x in data[lastUpdated:currentIndex]]

def getUserVectors():
	row = []
	col = []
	data = []
	imputationConstant = 3
	
	numRows = 0
	numMovies = 0
	lastUpdatedIndex = 0
	currentIndex = 0
	f = open("HW4_data/dev.queries")
	for line in iter(f):
		userData = line.split()
		userID = int(userData[0])
		
		sumOfSquares = 0
		lastUpdatedIndex = currentIndex
		for rating in userData[1:]:
			movieID = int(rating.split(":")[0])
			rating = int(rating.split(":")[1]) - imputationConstant
			sumOfSquares += math.pow(rating, 2)
			
			if numMovies < movieID:
				numMovies = movieID

			row.append(userID)
			col.append(movieID)
			data.append(rating)
			currentIndex += 1
		
		if sumOfSquares != 0:
			normalizeMatrixRow(data, lastUpdatedIndex, currentIndex, sumOfSquares)
		
		if numRows < userID:
			numRows = userID
		
	numRows += 1
	numMovies += 1
	userVectors = csr_matrix((data, (row, col)), shape=(numRows, numMovies))
	
	return numRows, numMovies, userVectors

def findKNearestNeighbours(k, userID, userVectors, numUsers):
	currentUser = userVectors.getrow(userID)
	for i in range(numUsers):
		if i == userID:
			continue
		
		sampleUser = userVectors.getrow(i)