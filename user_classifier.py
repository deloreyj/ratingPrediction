from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

userTrainMatrix = []
userTrainLabels = []
userIds = []
MISSINGVALUE = 'N/A'
with open('user.txt', 'r') as users:
    next(users)
    for user in users:
        userData = user.strip('\n').split(',')
        userIds.append(userData[0])
        if MISSINGVALUE not in userData:
            if userData[1] == 'M':
                userData[1] = 1
            else:
                userData[1] = 0
            userData[2] = int(userData[2])
            userData[3] = int(userData[3])
            userTrainMatrix.append(userData[1:2])
            userTrainLabels.append(userData[3])

train_split, test_split, train_split_labels, test_split_labels = train_test_split(userTrainMatrix, userTrainLabels, test_size=0.2)

# """
# Train the classifiers on a single split of the training data and compute accuracy
# """

dt = tree.DecisionTreeClassifier()
dt.fit(train_split, train_split_labels)
dt_score = dt.score(test_split, test_split_labels)
print "Accuracy of Decision Tree classifier: ", dt_score

rf = RandomForestClassifier()
rf.fit(train_split, train_split_labels)
rf_score = rf.score(test_split, test_split_labels)
print "Accuracy of Random Forest classifier: ", rf_score

gnb = GaussianNB()
gnb.fit(train_split, train_split_labels)
gnb_score = gnb.score(test_split, test_split_labels)
print "Accuracy of Gaussian Naive Bayes classifier: ", gnb_score

bnb = BernoulliNB()
bnb.fit(train_split, train_split_labels)
bnb_score = bnb.score(test_split, test_split_labels)
print "Accuracy of Bernoulli Naive Bayes classifier: ", bnb_score

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_split, train_split_labels)
knn_score = knn.score(test_split, test_split_labels)
print "Accuracy of K Nearest Neighbors classifier: ", knn_score

cv_dt = tree.DecisionTreeClassifier()
cv_dt_scores = cross_val_score(cv_dt, userTrainMatrix, userTrainLabels, cv=5)
print("Cross-Validated Accuracy of Decision Tree: %0.2f (+/- %0.2f)" % (cv_dt_scores.mean(), cv_dt_scores.std() * 2))

cv_rf = RandomForestClassifier()
cv_rf_scores = cross_val_score(cv_rf, userTrainMatrix, userTrainLabels, cv=5)
print("Cross-Validated Accuracy of Random Forest: %0.2f (+/- %0.2f)" % (cv_rf_scores.mean(), cv_rf_scores.std() * 2))

cv_gnb = GaussianNB()
cv_gnb_scores = cross_val_score(cv_gnb, userTrainMatrix, userTrainLabels, cv=5)
print("Cross-Validated Accuracy of Gaussian Naive Bayes: %0.2f (+/- %0.2f)" % (cv_gnb_scores.mean(), cv_gnb_scores.std() * 2))

cv_bnb = BernoulliNB()
cv_bnb_scores = cross_val_score(cv_bnb, userTrainMatrix, userTrainLabels, cv=5)
print("Cross-Validated Accuracy of Bernoulli Naive Bayes: %0.2f (+/- %0.2f)" % (cv_bnb_scores.mean(), cv_bnb_scores.std() * 2))

cv_knn = KNeighborsClassifier(n_neighbors=5)
cv_knn_scores = cross_val_score(cv_knn, userTrainMatrix, userTrainLabels, cv=5)
print("Cross-Validated Accuracy of K Nearest Neighbors: %0.2f (+/- %0.2f)" % (cv_knn_scores.mean(), cv_knn_scores.std() * 2))
