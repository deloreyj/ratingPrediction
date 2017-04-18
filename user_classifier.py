from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import copy

genderTrainMatrix = []
genderTrainLabels = []
ageTrainMatrix = []
ageTrainLabels = []
dataSet = []
MISSINGVALUE = 'N/A'
GENDERINDEX = 1
AGEINDEX = 2
OCCUPATIONINDEX = 3
predictGenderMatrix = []
predictGenderIndexes = []
predictGenderLabels = []
predictAgeMatrix = []
predictAgeIndexes = []
predictAgeLabels = []

def testClassifiers(trainMatrix, trainLabels, testMatrix, testLabels):
    # """
    # Train the classifiers on a single split of the training data and compute accuracy
    # """

    dt = tree.DecisionTreeClassifier()
    dt.fit(trainMatrix, trainLabels)
    dt_score = dt.score(testMatrix, testLabels)
    print "Accuracy of Decision Tree classifier: ", dt_score

    rf = RandomForestClassifier()
    rf.fit(trainMatrix, trainLabels)
    rf_score = rf.score(testMatrix, testLabels)
    print "Accuracy of Random Forest classifier: ", rf_score

    gnb = GaussianNB()
    gnb.fit(trainMatrix, trainLabels)
    gnb_score = gnb.score(testMatrix, testLabels)
    print "Accuracy of Gaussian Naive Bayes classifier: ", gnb_score

    bnb = BernoulliNB()
    bnb.fit(trainMatrix, trainLabels)
    bnb_score = bnb.score(testMatrix, testLabels)
    print "Accuracy of Bernoulli Naive Bayes classifier: ", bnb_score

def formatRow(row):
    newRow = copy.copy(row)
    newRow[0] = int(newRow[0])
    if newRow[1] == 'M':
        newRow[1] = 1
    elif newRow[1] == 'F':
        newRow[1] = 0
    if newRow[2] != MISSINGVALUE:
        newRow[2] = int(newRow[2])
    if newRow[3] != MISSINGVALUE:
        newRow[3] = int(newRow[3])
    return newRow


with open('user.txt', 'r') as users:
    next(users)
    for user in users:
        userData = user.strip('\n').split(',')
        dataSet.append(userData)
        if MISSINGVALUE not in userData:
            row = formatRow(userData)
            genderTrainMatrix.append(row[2:])
            genderTrainLabels.append(row[1])
        if MISSINGVALUE == userData[1] and MISSINGVALUE not in userData[2:]:
            row = formatRow(userData)
            predictGenderIndexes.append(row[0])
            predictGenderMatrix.append(row[2:])

train_split, test_split, train_split_labels, test_split_labels = train_test_split(genderTrainMatrix, genderTrainLabels, test_size=0.2)
testClassifiers(train_split, train_split_labels, test_split, test_split_labels)

# Predict gender values from age and occupation
rf = RandomForestClassifier()
rf.fit(genderTrainMatrix, genderTrainLabels)
predictGenderLabels = rf.predict(predictGenderMatrix)

# Replace gender values by index
for idx in range(len(predictGenderIndexes)):
    if predictGenderLabels[idx] == 1:
        dataSet[predictGenderIndexes[idx]][1] = 'M'
    else:
        dataSet[predictGenderIndexes[idx]][1] = 'F'

# Predict age from occupation and gender
for datum in dataSet:
    if MISSINGVALUE not in datum:
        row = formatRow(datum)
        ageTrainMatrix.append([row[1],row[3]])
        ageTrainLabels.append(row[2])
    if MISSINGVALUE == datum[2] and MISSINGVALUE not in [datum[1],datum[3]]:
        row = formatRow(datum)
        predictAgeIndexes.append(row[0])
        predictAgeMatrix.append([row[1],row[3]])

train_split, test_split, train_split_labels, test_split_labels = train_test_split(ageTrainMatrix, ageTrainLabels, test_size=0.2)
testClassifiers(train_split, train_split_labels, test_split, test_split_labels)

# Predict age from gender and occupation
rf = RandomForestClassifier()
rf.fit(ageTrainMatrix, ageTrainLabels)
predictAgeLabels = rf.predict(predictAgeMatrix)

# Replace gender values by index
for idx in range(len(predictAgeIndexes)):
    dataSet[predictAgeIndexes[idx]][2] = predictAgeLabels[idx]
print "predicted"

userOutput = open('userOutput.txt', 'w')
userOutput.write("Id,Gender,Age,Occupation\n")
for datum in dataSet:
    userOutput.write("{},{},{},{}\n".format(datum[0],datum[1],datum[2],datum[3]))
userOutput.close()


