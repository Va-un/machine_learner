# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# Load dataset
# here we are loading a data set irs so that we can work on it
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length',   'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url,   names=names)  # this command bring the data into compiler

# checking the contents
# shape
print(dataset.shape)  # It tells how many rows and column are there
# head
print(dataset.head(20))  # it shows the first x sets of the file here first 20 data
# descriptions
print(dataset.describe())  # it gives the basic information of the  data like mean,count,etc
# class distribution
print(dataset.groupby('class').size())  # it groups  that column inside data and shows it

# plotting
# box and whisker plots
dataset.plot(kind='box',   subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()  # gives box and wisker plot

# histograms
dataset.hist()
pyplot.show()  # will give histogram

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()  # pyplot matrix shall be shown

# Split-out test dataset
# here we are splitting thr data as train and test  usually the split is like 8:2 usually for better accuracy
# test>>train
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_test, Y_train,   Y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# train_test_split gives 4 output which is given variables like this.


# Spot Check Algorithms

models = []  # creating an list

models.append(('LR',   LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA',   LinearDiscriminantAnalysis()))
models.append(('KNN',   KNeighborsClassifier()))
models.append(('CART',   DecisionTreeClassifier()))
models.append(('NB',   GaussianNB()))
models.append(('SVM',   SVC(gamma='auto')))

# evaluate each model in   turn

results = []  # made a resultant dictionary

names = []   # made names list

for name, model in models:

    kfold = StratifiedKFold(n_splits=10, random_state=1,   shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train,   cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
print('%s: %f (%f)' % (name, cv_results.mean(),   cv_results.std()))

# Compare Algorithms
pyplot.boxplot(results,   labels=names)
pyplot.title('Algorithm   Comparison')
pyplot.show()  # this shows above data in a graph form for better reading

# From above graph we where able to tell that SVM Model would be best so we use SVM model


# Now we are giving the model train data and then testing it if it is able  predict
model = SVC()
model.fit(X_train,   Y_train)
predictions = model.predict(X_test)

# Now checking if the work done was right
# Evaluate predictions

print(accuracy_score(Y_test,   predictions))
print(confusion_matrix(Y_test,   predictions))
print(classification_report(Y_test,   predictions))