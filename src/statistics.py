from scipy.sparse import csr_matrix

f = open("data/train.csv")
numUsers = 0
numMovies = 0
sumOfRatings = 0
numRatings = 0
num1Rating = 0
num2Rating = 0
num3Rating = 0
num4Rating = 0
num5Rating = 0

numMoviesRatedByUser = 0
sumOfRatingsByUser = 0
num1RatingByUser = 0
num2RatingByUser = 0
num3RatingByUser = 0
num4RatingByUser = 0
num5RatingByUser = 0

numUsersRatedMovie = 0
sumOfRatingsByMovie = 0
num1RatingByMovie = 0
num2RatingByMovie = 0
num3RatingByMovie = 0
num4RatingByMovie = 0
num5RatingByMovie = 0

row = []
col = []
data = []

for line in iter(f):
	element = line.split(",")
	movieID = int(element[0])
	userID = int(element[1])
	rating = int(element[2])
	
	row.append(userID)
	col.append(movieID)
	data.append(rating)
	
	if rating == 1:
		num1Rating += 1
		if userID == 4321:
			num1RatingByUser += 1
		if movieID == 3:
			num1RatingByMovie += 1
	elif rating == 2:
		num2Rating += 1
		if userID == 4321:
			num2RatingByUser += 1
		if movieID == 3:
			num2RatingByMovie += 1
	elif rating == 3:
		num3Rating += 1
		if userID == 4321:
			num3RatingByUser += 1
		if movieID == 3:
			num3RatingByMovie += 1
	elif rating == 4:
		num4Rating += 1
		if userID == 4321:
			num4RatingByUser += 1
		if movieID == 3:
			num4RatingByMovie += 1
	elif rating == 5:
		num5Rating += 1
		if userID == 4321:
			num5RatingByUser += 1
		if movieID == 3:
			num5RatingByMovie += 1
	
	if userID == 4321:
		numMoviesRatedByUser += 1
		sumOfRatingsByUser += rating
	if movieID == 3:
		numUsersRatedMovie += 1
		sumOfRatingsByMovie += rating
	
	sumOfRatings += rating
	numRatings += 1
	numUsers = max(userID, numUsers)
	numMovies = max(movieID, numMovies)
f.close()
numUsers += 1
numMovies += 1
userVectors = csr_matrix((data, (row, col)), shape=(numUsers, numMovies))

print "Num Users: " + str(numUsers)
print "Num Movies: " + str(numMovies)
print "Avg Rating: " + str((sumOfRatings+0.0)/numRatings)
print "Number of Ratings 1: " + str(num1Rating)
print "Number of Ratings 2: " + str(num2Rating)
print "Number of Ratings 3: " + str(num3Rating)
print "Number of Ratings 4: " + str(num4Rating)
print "Number of Ratings 5: " + str(num5Rating)

print "\nFor User 4321:"
print "Number of Movies Rated: " + str(numMoviesRatedByUser)
print "Avg Rating: " + str((sumOfRatingsByUser+0.0)/numMoviesRatedByUser)
print "Number of Ratings 1: " + str(num1RatingByUser)
print "Number of Ratings 2: " + str(num2RatingByUser)
print "Number of Ratings 3: " + str(num3RatingByUser)
print "Number of Ratings 4: " + str(num4RatingByUser)
print "Number of Ratings 5: " + str(num5RatingByUser)

print "\nFor Movie 3:"
print "Number of Users Rated the Movie: " + str(numUsersRatedMovie)
print "Avg Rating: " + str((sumOfRatingsByMovie+0.0)/numUsersRatedMovie)
print "Number of Ratings 1: " + str(num1RatingByMovie)
print "Number of Ratings 2: " + str(num2RatingByMovie)
print "Number of Ratings 3: " + str(num3RatingByMovie)
print "Number of Ratings 4: " + str(num4RatingByMovie)
print "Number of Ratings 5: " + str(num5RatingByMovie)
