from sklearn import svm, datasets
import pickle

clf = svm.SVC()

iris = datasets.load_iris()
X, y = iris.data, iris.target

# understanding dataset shape
print X.shape
print str(X.shape[0]) + " observations of irises"
print str(X.shape[1]) + " features: sepal and petal length/width"

# train with all data
clf.fit(X, y)

# test with last item in dataset
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
prediction = clf2.predict(X[:1])
correct = y[0]

print prediction == correct
