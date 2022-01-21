import warnings
import pandas as pd
import numpy as np


warnings. filterwarnings("ignore")


df = pd.read_csv(r"weatherAUS (1).csv")
print('Size of weather data frame is :', df.shape)


df.head()


df.describe()

col_names = df.columns

col_names

df.count().sort_values()

df = df.drop(columns=['Sunshine','Evaporation','Cloud3pm','Cloud9am','Date'],axis=1)
df.shape

df = df.dropna()
df.shape

from sklearn import preprocessing

numerical = [var for var in df.columns if df[var].dtype=='float64']

for col in numerical:
    df[col] = preprocessing.scale(df[col])
    
df.head()



z = np.abs(stats.zscore(df._get_numeric_data()))
print(z)
df= df[(z < 3).all(axis = 1 )]
print(df.shape)

categorical = [var for var in df.columns if df[var].dtype=='object']

print("Number of categorical variables: ", len(categorical))

print(categorical)

df['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)
df['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)

categorical_columns = ['Location','WindGustDir', 'WindDir3pm', 'WindDir9am']




for col in categorical_columns:
    print(np.unique(df[col]))

df = pd.get_dummies(df, columns=categorical_columns)



from sklearn.model_selection import train_test_split



X = df.loc[:,df.columns!='RainTomorrow']

y = df.RainTomorrow


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
k_list = list(range(1,50,2))

cv_scores = []
from sklearn.model_selection import cross_val_score

for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)

plt.show()


best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)

from sklearn.neighbors import KNeighborsClassifier
k = 23
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(x_train, y_train)
preds = knn.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
accuracy = accuracy_score(y_test, preds)*100
print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

