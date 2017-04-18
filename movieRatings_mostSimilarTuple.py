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
import math, copy, random

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

    # For all possible genres, fill in a 1 if a movie contains that genre
    if genre == MISSING_VALUE:
        genre = [-1] * len(uniqGenres)
    else:
        genre = [1 if x in genre else 0 for x in uniqGenres]

    featureMatrix.append([gender, age, occupation, year] + genre)

"""
From a training/test file, reads each transaction, expands the ID fields
to store the features of the corresponding user/movie and appends the feature
vector to the overall feature matrix.
"""
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
with open('user.txt', 'r') as users:
    next(users)
    for user in users:
        userData = user.strip('\n').split(',')
        userID = userData[0]
        usersData[userID] = userData[1:]

def computeDistance(list1, list2):
    return (sum([(x - y) ** 2 for x, y in zip(list1, list2)]))**0.5

def closestNeighbor(orig_list, featureMatrix):
    minDistance = 99999;
    bestMatch = []
    numReps = 0

    while True:
        neighbor = featureMatrix[random.randint(0, len(featureMatrix) - 1)]
        if -1 in neighbor:
            continue
        dist = computeDistance(orig_list, neighbor)
        numReps += 1
        if dist < minDistance:
            minDistance = dist
            bestMatch = neighbor
            numReps = 0

        if numReps == 500:
            break

    return copy.deepcopy(bestMatch)

# Read and store the training and test feature matrices
trainFeatureMatrix, trainLabels = extractDataAndLabels('train.txt', usersData, moviesData, list(genreSet))
testFeatureMatrix, testIds = extractDataAndLabels('test.txt', usersData, moviesData, list(genreSet))

# Combine train and test matrices so better matches can be found
allFeatureMatrix = copy.deepcopy(trainFeatureMatrix)
for testSample in testFeatureMatrix:
    allFeatureMatrix.append(testSample)

missingCount = 0
for sample in trainFeatureMatrix:
    if -1 in sample:
        missingCount += 1
        sample = closestNeighbor(sample, trainFeatureMatrix)
        # Progress bar, will finish at 336000
        if missingCount % 1000 == 0:
            print missingCount

print "done train"

for sample in testFeatureMatrix:
    if -1 in sample:
        sample = closestNeighbor(sample, allFeatureMatrix)

print "done test"

train_split, test_split, train_split_labels, test_split_labels = train_test_split(trainFeatureMatrix, trainLabels, test_size=0.15)

bnb = BernoulliNB()
bnb.fit(train_split, train_split_labels)
bnb_score = bnb.score(test_split, test_split_labels)
print "Accuracy of Bernoulli Naive Bayes classifier: ", bnb_score

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

# bnb = BernoulliNB()
# bnb.fit(train_split, train_split_labels)
# bnb_score = bnb.score(test_split, test_split_labels)
# print "Accuracy of Bernoulli Naive Bayes classifier: ", bnb_score

# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(train_split, train_split_labels)
# knn_score = knn.score(test_split, test_split_labels)
# print "Accuracy of K Nearest Neighbors classifier: ", knn_score


# For now, output the test labels using a Bernoulli Naive Bayes Classifier
predTestLabels = bnb.predict(testFeatureMatrix)

testOutput = open('testOutput.txt', 'w')
testOutput.write("Id,rating\n")
for testId, label in zip(testIds, predTestLabels):
    testOutput.write("{},{}\n".format(testId, label))
testOutput.close()