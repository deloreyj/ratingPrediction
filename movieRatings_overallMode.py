"""
Performs classification to determine a user's rating for a movie.

Given a movie file with attributes and ratings and a user file, train
several classifiers on these datasets in order to compute a users' ratings
for a test movie set.

Authors: "James Delorey and Siva Mullapudi"
Version: Midterm Checkpoint
"""

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import copy
import operator

MISSING_VALUE = "N/A"


"""
Converts elements of a transaction and adds them to the feature matrix.
For now, missing values are given a placeholder -1 value.
"""
def addTransaction(transactionData, featureMatrix, uniqGenres):
    gender = transactionData[0]
    age = transactionData[1]
    occupation = transactionData[2]
    year = transactionData[3]
    genre = transactionData[4]

    # Convert Male/Female to binary value, -1 if missing
    if gender == "M":
        gender = 2
    elif gender == "F":
        gender = 1
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

    # For all possible genres, fill in a 1 if a movie contains that genre
    if genre == MISSING_VALUE:
        genre = [0] * len(uniqGenres)
    else:
        genre = [1 if x in genre else 0 for x in uniqGenres]

    featureMatrix.append([gender, age, occupation, year] + genre)

def replaceMissingValuesAndAddTransaction(userData, movieData, overallMode, featureMatrix, uniqGenres):
    gender = userData[0]
    age = userData[1]
    occupation = userData[2]
    year = movieData[0]
    genre = movieData[1]

    if gender == "M":
        gender = 2
    elif gender == "F":
        gender = 1
    else:
        gender = overallMode['gender']

    if age == MISSING_VALUE:
        age = overallMode['age']
    else:
        age = int(age)

    if occupation == MISSING_VALUE:
        occupation = overallMode['occupation']
    else:
        occupation = int(occupation)

    if year == MISSING_VALUE:
        year = overallMode['year']
    else:
        year = int(year)

    # For all possible genres, fill in a 1 if a movie contains that genre
    if genre == MISSING_VALUE:
        genre = [1 if x in overallMode['genre'] else 0 for x in uniqGenres]
    else:
        genre = [1 if x in genre else 0 for x in uniqGenres]

    featureMatrix.append([gender, age, occupation, year] + genre)


"""
From a training/test file, reads each transaction, expands the ID fields
to store the features of the corresponding user/movie and appends the feature
vector to the overall feature matrix.
"""
def extractDataAndLabels(filename, usersData, moviesData, uniqGenres, overallMode):
    featureMatrix = [[]]
    labels = []
    testIds = []
    with open(filename, 'r') as dataset:
        next(dataset)
        for transaction in dataset:
            transactionData = transaction.strip('\n').split(',')
            transactionMovieData = moviesData[transactionData[2]]
            transactionUserData = usersData[transactionData[1]]
            if filename == "train.txt":
                label = transactionData[3]
                labels.append(label)
                # replaceMissingValuesAndAddTransaction(transactionUserData, transactionMovieData, overallMode, featureMatrix, uniqGenres)
            else:
                testId = transactionData[0]
                testIds.append(testId)
                # addTransaction(transactionUserData + transactionMovieData, featureMatrix, uniqGenres)

            replaceMissingValuesAndAddTransaction(transactionUserData, transactionMovieData, overallMode,
                                                  featureMatrix, uniqGenres)
    # Delete the header line
    featureMatrix.pop(0)

    if filename == "train.txt":
        return featureMatrix, labels
    else:
        return featureMatrix, testIds

"""
    Main Function
"""

# Read the movies file and store a dictionary of the features
moviesData = {}
genreSet = set()
with open('movie.txt', 'r') as movies:
    next(movies)
    for movie in movies:
        movieData = movie.strip('\n').split(',')
        movieID = movieData[0]
        moviesData[movieID] = movieData[1:]
        genreSet.update(movieData[-1].split('|'))

# Read the users file and store a dictionary of the features
usersData = {}
with open('userOutput.txt', 'r') as users:
    next(users)
    for user in users:
        userData = user.strip('\n').split(',')
        userID = userData[0]
        usersData[userID] = userData[1:]

transactionDict = {
    'gender': {},
    'age': {},
    'occupation': {},
    'year': {},
    'genre': {}
}

overallCounts = copy.deepcopy(transactionDict) #[]
overallMode = copy.deepcopy(transactionDict) #[]
# for i in range(5):
#     labelCounts.append(copy.deepcopy(transactionDict))
#     labelMeans.append(copy.deepcopy(transactionDict))

with open("train.txt", 'r') as dataset:
    next(dataset)
    for transaction in dataset:
        transactionData = transaction.strip('\n').split(',')
        label = int(transactionData[3])
        transactionMovieData = moviesData[transactionData[2]]
        transactionUserData = usersData[transactionData[1]]

        if transactionUserData[0] in overallCounts['gender']:
            # labelCounts[label - 1]['gender'][transactionUserData[0]] += 1
            overallCounts['gender'][transactionUserData[0]] += 1
        else:
            overallCounts['gender'][transactionUserData[0]] = 1

        if transactionUserData[1] in overallCounts['age']:
            overallCounts['age'][transactionUserData[1]] += 1
        else:
            overallCounts['age'][transactionUserData[1]] = 1

        if transactionUserData[2] in overallCounts['occupation']:
            overallCounts['occupation'][transactionUserData[2]] += 1
        else:
            overallCounts['occupation'][transactionUserData[2]] = 1

        if transactionMovieData[0] in overallCounts['year']:
            overallCounts['year'][transactionMovieData[0]] += 1
        else:
            overallCounts['year'][transactionMovieData[0]] = 1

        if transactionMovieData[1] in overallCounts['genre']:
            overallCounts['genre'][transactionMovieData[1]] += 1
        else:
            overallCounts['genre'][transactionMovieData[1]] = 1

overallCounts['gender'].pop('N/A', None)
overallCounts['age'].pop('N/A', None)
overallCounts['occupation'].pop('N/A', None)
overallCounts['year'].pop('N/A', None)
overallCounts['genre'].pop('N/A', None)

overallMode['gender'] = max(overallCounts['gender'].iteritems(), key=operator.itemgetter(1))[0]
overallMode['age'] = int(max(overallCounts['age'].iteritems(), key=operator.itemgetter(1))[0])
overallMode['occupation'] = int(max(overallCounts['occupation'].iteritems(), key=operator.itemgetter(1))[0])
overallMode['year'] = int(max(overallCounts['year'].iteritems(), key=operator.itemgetter(1))[0])
overallMode['genre'] = max(overallCounts['genre'].iteritems(), key=operator.itemgetter(1))[0]
if overallMode['gender'] == 'M':
    overallMode['gender'] = 1
else:
    overallMode['gender'] = 2


# Read and store the training and test feature matrices
trainFeatureMatrix, trainLabels = extractDataAndLabels('train.txt', usersData, moviesData, list(genreSet), overallMode)
testFeatureMatrix, testIds = extractDataAndLabels('test.txt', usersData, moviesData, list(genreSet), overallMode)

# print trainFeatureMatrix[0]
# print trainFeatureMatrix[1]

train_split, test_split, train_split_labels, test_split_labels = train_test_split(trainFeatureMatrix, trainLabels, test_size=0.2)


"""
Train the classifiers on a single split of the training data and compute accuracy
"""

# dt = tree.DecisionTreeClassifier()
# dt.fit(train_split, train_split_labels)
# dt_score = dt.score(test_split, test_split_labels)
# print "Accuracy of Decision Tree classifier: ", dt_score
#
# rf = RandomForestClassifier()
# rf.fit(train_split, train_split_labels)
# rf_score = rf.score(test_split, test_split_labels)
# print "Accuracy of Random Forest classifier: ", rf_score
#
# gnb = GaussianNB()
# gnb.fit(train_split, train_split_labels)
# gnb_score = gnb.score(test_split, test_split_labels)
# print "Accuracy of Gaussian Naive Bayes classifier: ", gnb_score
#
bnb = BernoulliNB()
bnb.fit(train_split, train_split_labels)
bnb_score = bnb.score(test_split, test_split_labels)
print "Accuracy of Bernoulli Naive Bayes classifier: ", bnb_score
#
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(train_split, train_split_labels)
# knn_score = knn.score(test_split, test_split_labels)
# print "Accuracy of K Nearest Neighbors classifier: ", knn_score

"""
Train the classifiers with a cross-validated scheme on the whole training set
"""

# cv_dt = tree.DecisionTreeClassifier()
# cv_dt_scores = cross_val_score(cv_dt, trainFeatureMatrix, trainLabels, cv=5)
# print("Cross-Validated Accuracy of Decision Tree: %0.2f (+/- %0.2f)" % (cv_dt_scores.mean(), cv_dt_scores.std() * 2))
#
# cv_rf = RandomForestClassifier()
# cv_rf_scores = cross_val_score(cv_rf, trainFeatureMatrix, trainLabels, cv=5)
# print("Cross-Validated Accuracy of Random Forest: %0.2f (+/- %0.2f)" % (cv_rf_scores.mean(), cv_rf_scores.std() * 2))
#
# cv_gnb = GaussianNB()
# cv_gnb_scores = cross_val_score(cv_gnb, trainFeatureMatrix, trainLabels, cv=5)
# print("Cross-Validated Accuracy of Gaussian Naive Bayes: %0.2f (+/- %0.2f)" % (cv_gnb_scores.mean(), cv_gnb_scores.std() * 2))
#
# cv_bnb = BernoulliNB()
# cv_bnb_scores = cross_val_score(cv_bnb, trainFeatureMatrix, trainLabels, cv=5)
# print("Cross-Validated Accuracy of Bernoulli Naive Bayes: %0.2f (+/- %0.2f)" % (cv_bnb_scores.mean(), cv_bnb_scores.std() * 2))
#
# cv_knn = KNeighborsClassifier(n_neighbors=5)
# cv_knn_scores = cross_val_score(cv_knn, trainFeatureMatrix, trainLabels, cv=5)
# print("Cross-Validated Accuracy of K Nearest Neighbors: %0.2f (+/- %0.2f)" % (cv_knn_scores.mean(), cv_knn_scores.std() * 2))


# For now, output the test labels using a Bernoulli Naive Bayes Classifier
predTestLabels = bnb.predict(testFeatureMatrix)

testOutput = open('testOutput.txt', 'w')
testOutput.write("Id,rating\n")
for testId, label in zip(testIds, predTestLabels):
    testOutput.write("{},{}\n".format(testId, label))
testOutput.close()