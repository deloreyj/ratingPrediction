import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

MISSING_VALUE = "N/A"

def addTransaction(transactionData, trainFeatureMatrix, uniqGenres):
    gender = transactionData[0]
    age = transactionData[1]
    occupation = transactionData[2]
    year = transactionData[3]
    genre = transactionData[4]

    # Convert male/female to binary value
    if gender == "M":
        gender = 1
    elif gender == "F":
        gender = 0
    else:
        gender = -1

    if age == MISSING_VALUE:
        age = -1
    else:
        age = int(age)
    if occupation == MISSING_VALUE:
        occupation = -1
    else:
        occupation = int(occupation)

    if year == MISSING_VALUE:
        year = -1
    else:
        year = int(year)

    # For now, just put a hash value of the genre string
    if genre == MISSING_VALUE:
        genre = [0] * len(uniqGenres)
    else:
        genre = [1 if x in genre else 0 for x in uniqGenres]

    trainFeatureMatrix.append([gender, age, occupation, year] + genre)


moviesData = {}
genreSet = set()
with open('movie.txt', 'r') as movies:
    next(movies)
    for movie in movies:
        movieData = movie.strip('\n').split(',')
        movieID = movieData[0]
        moviesData[movieID] = movieData[1:]
        genreSet.update(movieData[-1].split('|'))

usersData = {}

with open('user.txt', 'r') as users:
    next(users)
    for user in users:
        userData = user.strip('\n').split(',')
        userID = userData[0]
        usersData[userID] = userData[1:]

trainFeatureMatrix = [[]]
trainLabels = []
uniqGenres = list(genreSet)
with open('train.txt', 'r') as trainData:
    next(trainData)
    for transaction in trainData:
        transactionData = transaction.strip('\n').split(',')
        trainLabel = transactionData[3]
        trainLabels.append(trainLabel)

        transactionMovieData = moviesData[transactionData[2]]
        transactionUserData = usersData[transactionData[1]]
        addTransaction(transactionUserData + transactionMovieData, trainFeatureMatrix, uniqGenres)
        # trainFeatureMatrix.append(transactionUserData + transactionMovieData)
# Delete the header line
trainFeatureMatrix.pop(0)

train, test, train_labels, test_labels = train_test_split(trainFeatureMatrix, trainLabels, test_size=0.2)

dt = tree.DecisionTreeClassifier()
dt.fit(train, train_labels)
dt_score = dt.score(test, test_labels)

print "Accuracy of Decision Tree classifier: ", dt_score

rf = RandomForestClassifier()
rf.fit(train, train_labels)
rf_score = rf.score(test, test_labels)

print "Accuracy of Random Forest classifier: ", rf_score

gnb = GaussianNB()
gnb.fit(train, train_labels)
gnb_score = gnb.score(test, test_labels)

print "Accuracy of Gaussian Naive Bayes classifier: ", gnb_score

bnb = BernoulliNB()
bnb.fit(train, train_labels)
bnb_score = bnb.score(test, test_labels)

print "Accuracy of Bernoulli Naive Bayes classifier: ", bnb_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train, train_labels)
knn_score = knn.score(test, test_labels)

print "Accuracy of K Nearest Neighbors classifier: ", knn_score


# print trainFeatureMatrix[0]
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