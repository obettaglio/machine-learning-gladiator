from sklearn import svm, datasets
import pickle

clf = svm.SVC()

iris = datasets.load_iris()
X, y = iris.data, iris.target

clf.fit(X, y)

s = pickle.dumps(clf)
clf2 = pickle.loads(s)
prediction = clf2.predict(X[:1])
correct = y[0]

print prediction == correct
