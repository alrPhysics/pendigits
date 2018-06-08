## Classify digits written on a WACOM PL-100V tablet that are expressed as 16 attributes

The [Pen-Based Recognition of Handwritten Digits Data Set](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) was used for this classification project. The data seems to have already been conditioned as I could not generate significantly better results using preprocessing techniques. Scaling the features and performing dimensionality reduction generated much worse results than when the raw data was used.

### sklearn ML Algorithms Used:
* KNeighborsClassifier
* LogisticRegression
* MLPClassifier

GridSearchCV was used to optimize the KNeighbors and MLPClassifier models
