from sklearn import svm, datasets

digits = datasets.load_digits()

classifier = svm.SVC(gamma=0.001, C=100.)

classifier.fit(digits.data[:-1], digits.target[:-1])
prediction = classifier.predict(digits.data[-1:])

print prediction
