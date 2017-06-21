from sklearn.cross_validation import cross_val_score
scores = cross_val_score(svm, Xtask, ytask, cv=10)
print(scores)

