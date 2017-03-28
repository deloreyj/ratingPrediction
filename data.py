import numpy as np
from sklearn import tree


moviesData = {}

with open('movie.txt', 'r') as movies:
    for movie in movies:
        movieData = movie.strip('\n').split(',')
        movieID = movieData[0]
        moviesData[movieID] = movieData[1:]

usersData = {}

with open('user.txt', 'r') as users:
    for user in users:
        userData = user.strip('\n').split(',')
        userID = userData[0]
        usersData[userID] = userData[1:]

trainFeatureData = [[]]
trainLabels = []
with open('train.txt', 'r') as trainData:
    next(trainData)
    for transaction in trainData:
        transactionData = transaction.strip('\n').split(',')
        trainLabel = transactionData[3]
        transactionMovieData = moviesData[transactionData[2]]
        transactionUserData = usersData[transactionData[1]]

        trainLabels.append(trainLabel)
        trainFeatureData.append(transactionUserData + transactionMovieData)
# Delete the header line
trainFeatureData.pop(0)

clf = tree.DecisionTreeClassifier()
clf.fit(trainFeatureData, trainLabels)

# print trainFeatureData[0]
# print trainLabels[0]

# users = open('user.txt', 'r')

# movieMatrix = np.fromfile('movie.txt', dtype=float, count=-1, sep=",")
# movieData = np.genfromtxt('movie.txt', dtype=None, delimiter=",", skiprows=1)
# userData = np.genfromtxt('user.txt', dtype=None, delimiter=",", skiprows=1)
# print tuserData[0]
# print np.asarray(userData)

# with open('train.txt', 'r') as train_data:
#     for transaction in train_data:
#         movieData = movie.strip('\n').split(',')

# movieMatrix = [[]]
#
# moviesLength = 0
#
# with open('movie.txt', 'r') as movies:
#     for movie in movies:
#         movieData = movie.strip('\n').split(',')
#         movieMatrix.append(movieData[1:])
#
# movieMatrix.pop(0)
#
# print movieMatrix[0]
# print movieMatrix[1]

# movies.close()


# for movie in movies:
#     moviesLength += 1
#     splitMovie = movie.strip('\n').split(',')
#     if splitMovie[0] in movieDict['id']:
#         movieDict['id'][splitMovie[0]] += 1
#     else:
#         movieDict['id'][splitMovie[0]] = 1
#     if splitMovie[1] in movieDict['year']:
#         movieDict['year'][splitMovie[1]] += 1
#     else:
#         movieDict['year'][splitMovie[1]] = 1
#     if splitMovie[2] in movieDict['genre']:
#         movieDict['genre'][splitMovie[2]] += 1
#     else:
#         movieDict['genre'][splitMovie[2]] = 1
# print 'LIST LENGTH >>>>>: %s' % moviesLength
# print 'EMPTY YEARS >>>>>: %s, %s%%' % (movieDict['year']['N/A'], int(movieDict['year']['N/A']/float(moviesLength - 1) * 100))
# print 'EMPTY GENRES >>>>>: %s, %s%%' % (movieDict['genre']['N/A'], int(movieDict['genre']['N/A']/float(moviesLength - 1) * 100))
#
# w, h = 8, 5;
# num_features =
# num_train_samples =
# Matrix = [[0 for x in range(w)] for y in range(h)]