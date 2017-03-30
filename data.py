import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

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

    if genre == MISSING_VALUE:
        genre = [0] * len(uniqGenres)
    else:
        genre = [1 if x in genre else 0 for x in uniqGenres]

    trainFeatureMatrix.append([gender, age, occupation, year] + genre)

def extractDataAndLabels(filename, usersData, moviesData, uniqGenres):
    featureMatrix = [[]]
    labels = []
    testIds = []
    with open(filename, 'r') as dataset:
        next(dataset)
        for transaction in dataset:
            transactionData = transaction.strip('\n').split(',')
            if filename == "train.txt":
                label = transactionData[3]
                labels.append(label)
            else:
                testId = transactionData[0]
                testIds.append(testId)

            transactionMovieData = moviesData[transactionData[2]]
            transactionUserData = usersData[transactionData[1]]
            addTransaction(transactionUserData + transactionMovieData, featureMatrix, uniqGenres)
    # Delete the header line
    featureMatrix.pop(0)

    if filename == "train.txt":
        return featureMatrix, labels
    else:
        return featureMatrix, testIds


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

trainFeatureMatrix, trainLabels = extractDataAndLabels('train.txt', usersData, moviesData, list(genreSet))

train_split, test_split, train_split_labels, test_split_labels = train_test_split(trainFeatureMatrix, trainLabels, test_size=0.2)

# Decision Tree
dt = tree.DecisionTreeClassifier()
dt.fit(train_split, train_split_labels)
dt_score = dt.score(test_split, test_split_labels)
print "Accuracy of Decision Tree classifier: ", dt_score

cv_dt = tree.DecisionTreeClassifier()
cv_dt_scores = cross_val_score(cv_dt, trainFeatureMatrix, trainLabels, cv=5)
print("Cross-Validated Accuracy of Decision Tree: %0.2f (+/- %0.2f)" % (cv_dt_scores.mean(), cv_dt_scores.std() * 2))

# Random Forest
rf = RandomForestClassifier()
rf.fit(train_split, train_split_labels)
rf_score = rf.score(test_split, test_split_labels)
print "Accuracy of Random Forest classifier: ", rf_score

cv_rf = tree.DecisionTreeClassifier()
cv_rf_scores = cross_val_score(cv_rf, trainFeatureMatrix, trainLabels, cv=5)
print("Cross-Validated Accuracy of Random Forest: %0.2f (+/- %0.2f)" % (cv_rf_scores.mean(), cv_rf_scores.std() * 2))

# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(train_split, train_split_labels)
gnb_score = gnb.score(test_split, test_split_labels)
print "Accuracy of Gaussian Naive Bayes classifier: ", gnb_score

cv_gnb = GaussianNB()
cv_gnb_scores = cross_val_score(cv_gnb, trainFeatureMatrix, trainLabels, cv=5)
print("Cross-Validated Accuracy of Gaussian Naive Bayes: %0.2f (+/- %0.2f)" % (cv_gnb_scores.mean(), cv_gnb_scores.std() * 2))

# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(train_split, train_split_labels)
bnb_score = bnb.score(test_split, test_split_labels)
print "Accuracy of Bernoulli Naive Bayes classifier: ", bnb_score

cv_bnb = BernoulliNB()
cv_bnb_scores = cross_val_score(cv_bnb, trainFeatureMatrix, trainLabels, cv=5)
print("Cross-Validated Accuracy of Bernoulli Naive Bayes: %0.2f (+/- %0.2f)" % (cv_bnb_scores.mean(), cv_bnb_scores.std() * 2))

# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_split, train_split_labels)
knn_score = knn.score(test_split, test_split_labels)
print "Accuracy of K Nearest Neighbors classifier: ", knn_score

cv_knn = KNeighborsClassifier(n_neighbors=5)
cv_knn_scores = cross_val_score(cv_knn, trainFeatureMatrix, trainLabels, cv=5)
print("Cross-Validated Accuracy of K Nearest Neighbors: %0.2f (+/- %0.2f)" % (cv_knn_scores.mean(), cv_knn_scores.std() * 2))

testFeatureMatrix, testIds = extractDataAndLabels('test.txt', usersData, moviesData, list(genreSet))

# For now, just output test results using Decision Tree Classifier
predTestLabels = dt.predict(testFeatureMatrix)

testOutput = open('testOutput.txt', 'w')
testOutput.write("Id,rating\n")
for testId, label in zip(testIds, predTestLabels):
    testOutput.write("{},{}\n".format(testId, label))
testOutput.close()

# users = open('user.txt', 'r')

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