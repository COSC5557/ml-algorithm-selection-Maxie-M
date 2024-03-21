## References:
##https://www.analyticsvidhya.com/blog/2022/02/logistic-regression-using-python-and-excel/
##https://medium.com/@draj0718/logistic-regression-with-standardscaler-from-the-scratch-ec01def674e8
##https://medium.com/@diehardankush/how-implementing-decision-trees-in-python-with-scikit-learn-part-3-29e5a787baaf


#importing needed tools 
%matplotlib inline
import numpy as np 
import pandas as pd 
import seaborn as sns
from scipy import stats 
from sklearn import tree
import matplotlib.pylab as plt
from sklearn import preprocessing 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 


#import red wine dataset 
red_wine_data = pd.read_csv('winequality-red.csv')


#shape of red wine dataframe 
red_wine_data.shape


#checking the data type of the red wine dataset 
red_wine_data.dtypes


#information on red wine dataset 
red_wine_data.info()


#first 5 rows of red wine dataframe 
red_wine_data.head()


#all columns in red wine dataframe 
red_wine_data.columns


#checking for missing values in red wine dataset 
red_wine_data.isnull().sum()


##no missing values##


#checking for any duplicates in red wine dataset 
red_wine_data.duplicated().sum()


#duplicates in the red_wine_data 
red_wine_data.loc[red_wine_data.duplicated(), :]


#dropping duplicates in dataset 
red_wine_data.drop_duplicates(inplace = True, keep = 'first')


#check to see if the duplicates were handled in the dataset 
red_wine_data.duplicated().sum()


#untouched red wine data
red_wine_df = red_wine_data[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
red_wine_untouched = sns.boxplot(data = red_wine_df, orient = "h", palette = "Set2")
plt.show()


#re-looking at dataset shape after dealing with duplicates 
red_wine_data.shape


#checking unique elements 
red_wine_data['quality'].unique()


#binarization of the target variable, using list comprehension
red_wine_data['quality'] = [1 if x>=7 else 0 for x in red_wine_data['quality']]
red_wine_data['quality'].unique()


#splitting into train and test   
X = red_wine_data.drop('quality' , axis = 1)
y = red_wine_data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


red_wine_data.info()


#Decision Tree Classifier
decTr_rw = DecisionTreeClassifier(random_state = 0, max_depth = 10)
decTr_rw.fit(X_train, y_train)


y_pred_decTr_rw = decTr_rw.predict(X_test)


#checking accuracy 
acc_decTr_rw = accuracy_score(y_test, y_pred_decTr_rw)
print('The Accuracy for Test Set is: {}'.format(acc_decTr_rw *100))


#creating conusion matrix for decTr 
con_mat_decTr_rw = confusion_matrix(y_test, y_pred_decTr_rw)
print(con_mat_decTr_rw)


#creating the heatmap DecisionTreeClassier() 
plt.figure(figsize = (12, 6))
plt.title('Confusion Matrix for Quality of Red Wine (DecisionTreeClassifer())')
sns.heatmap(con_mat_decTr_rw, annot = True, fmt = 'd', cmap = 'BuPu')
plt.ylabel('Actual Quality of Red Wine')
plt.xlabel('Predicted Quality of Red Wine')
plt.show()


#RandomForestClassier() 
ranFor_rw = RandomForestClassifier(random_state = 0, n_estimators=100, max_depth = 10)
ranFor_rw.fit(X_train, y_train)


y_pred_ranFor_rw = ranFor_rw.predict(X_test)


#checking accuracy 
acc_ranFor_rw = accuracy_score(y_test, y_pred_ranFor_rw)
print('The Accuracy for Test Set is: {}'.format(acc_ranFor_rw *100))


#creating confusion matrix for ranFor 
con_mat_ranFor_rw = confusion_matrix(y_test, y_pred_ranFor_rw)
print(con_mat_ranFor_rw)


#creating the heatmap RandomForestClassier() 
plt.figure(figsize = (12, 6))
plt.title('Confusion Matrix for Quality of Red Wine (RandomForestClassifer())')
sns.heatmap(con_mat_ranFor_rw, annot = True, fmt = 'd', cmap = 'BuPu')
plt.ylabel('Actual Quality of Red Wine')
plt.xlabel('Predicted Quality of Red Wine')
plt.show()


#Logistic Regression 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


rw_logReg = LogisticRegression(random_state = 0) 
rw_logReg.fit(X_train , y_train)


y_pred_logReg_rw = rw_logReg.predict(X_test)


#checking accuracy 
test_acc_logReg_rw = accuracy_score(y_test, y_pred_logReg_rw)
print('The Accuracy for Test Set is {}'.format(test_acc_logReg_rw * 100))


#creating confusion matrix for logistic red wine dataset 
con_mat_logReg_rw = confusion_matrix(y_test, y_pred_logReg_rw)
print(con_mat_logReg_rw)


#creating the heatmap logistic 
plt.figure(figsize = (12, 6))
plt.title('Confusion Matrix for Quality of Red Wine (Logistic Regression)')
sns.heatmap(con_mat_logReg_rw, annot = True, fmt = 'd', cmap = 'BuPu')
plt.ylabel('Actual Quality of Red Wine')
plt.xlabel('Predicted Quality of Red Wine')
plt.show()


#prep models 
models = []
models.append(('decTr', acc_decTr_rw))
models.append(('ranFor', acc_ranFor_rw))
models.append(('logReg', acc_logReg_rw))


#model names and accuracies 
names = [model[0] for model in models]
accuracy = [model[1] for model in models]


#visualizing 
plt.figure(figsize=(10,6))
bars = plt.bar(names, accuracy, color='Blue')
plt.title('Algorithm Comparisons')
plt.xlabel('Machine Learning Algorithms')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.yticks(np.arange(0,1.1,0.1))
plt.grid(axis='y', linestyle='--', alpha=0.7)


#adding the accuracy percentage on top of the bars
for bar, acc in zip(bars, accuracy):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{acc:.2f}', ha='center', va='bottom')


plt.show()
