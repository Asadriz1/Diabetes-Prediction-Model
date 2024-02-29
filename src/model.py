from sklearn import svm
from sklearn.metrics import accuracy_score

def train_svm(X_train, Y_train, kernel='linear'):
    classifier = svm.SVC(kernel=kernel)
    classifier.fit(X_train, Y_train)
    return classifier

def evaluate_model(classifier, X, Y):
    predictions = classifier.predict(X)
    accuracy = accuracy_score(predictions, Y)
    return accuracy
