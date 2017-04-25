"""
Performs classification to determine a user's rating for a movie.

Given a movie file with attributes and ratings and a user file, train
several classifiers on these datasets in order to compute a users' ratings
for a test movie set.

Authors: "James Delorey and Siva Mullapudi"
Version: Final Submission
"""

import numpy as np

MISSING_VALUE = "N/A"

"""
    Main Function
"""

maxUserId = 0
with open('user.txt', 'r') as users:
    next(users)
    for user in users:
        userData = user.strip('\n').split(',')
        userId = int(userData[0])
        if userId > maxUserId:
            maxUserId = userId

maxMovieId = 0
with open('movie.txt', 'r') as movies:
    next(movies)
    for movie in movies:
        movieData = movie.strip('\n').split(',')
        movieId = int(movieData[0])
        if movieId > maxMovieId:
            maxMovieId = movieId

userMovieRatings = np.zeros((maxUserId, maxMovieId))

with open('train.txt', 'r') as trainData:
    next(trainData)
    for transaction in trainData:
        transactionData = transaction.strip('\n').split(',')
        movieId = int(transactionData[2]) - 1
        userId = int(transactionData[1]) - 1
        label = int(transactionData[3])

        userMovieRatings[userId, movieId] = label

n_factors = 100
userMat = np.random.rand(maxUserId, n_factors)
movieMat = np.random.rand(n_factors, maxMovieId)

# initialize biases
userBias = np.zeros(maxUserId)
movieBias = np.zeros(maxMovieId)
overallBias = np.mean(userMovieRatings[np.where(userMovieRatings != 0)])

num_steps = 10
stepLen = 0.01
lam = 0.05
userRegularization = 0.01
movieRegularization = 0.01

nonZero_rows, nonZero_cols = np.nonzero(userMovieRatings)

for step in xrange(num_steps):
    for r, c in zip(nonZero_rows, nonZero_cols):
        predictionBias = overallBias + userBias[r] + movieBias[c]
        err = userMovieRatings[r, c] - (userMat[r, :].dot(movieMat[:, c]) + predictionBias)
        userMat[r, :] += stepLen * (err * movieMat[:, c] - lam * userMat[r, :])
        movieMat[:, c] += stepLen * (err * userMat[r, :] - lam * movieMat[:, c])

        #update biases
        userBias[r] += stepLen * (err - userRegularization * userBias[r])
        movieBias[c] += stepLen * (err - movieRegularization * movieBias[c])

    totalErr = 0
    for r, c in zip(nonZero_rows, nonZero_cols):
        totalErr += (userMovieRatings[r, c] - userMat[r, :].dot(movieMat[:, c])) ** 2

    print totalErr

testIds = []
predLabels = []
with open('test.txt', 'r') as testData:
    next(testData)
    for transaction in testData:
        transactionData = transaction.strip('\n').split(',')
        movieId = int(transactionData[2]) - 1
        userId = int(transactionData[1]) - 1
        predictionBias = overallBias + userBias[userId] + movieBias[movieId]
        predLabels.append(int(round(userMat[userId, :].dot(movieMat[:, movieId]) + predictionBias)))
        testIds.append(transactionData[0])

testOutput = open('testOutput.txt', 'w')
testOutput.write("Id,rating\n")
for testId, label in zip(testIds, predLabels):
    testOutput.write("{},{}\n".format(testId, label))
testOutput.close()
