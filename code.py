# --------------
# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# Code starts here

df = pd.read_json(path, lines = True)
#print(df.head())

df.columns = df.columns.str.lower().str.replace(' ', '_')
#print(df.head())

missing_data = df.isnull()
#print(missing_data.head())

df = df.drop(['waist', 'bust', 'user_name','review_text','review_summary','shoe_size','shoe_width'],axis =1)
#print(df.head())
X = df.drop(['fit'],1)
#print(X.head())
y = df['fit'].copy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 6)
#print(X_train.head())
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Code ends here


# --------------
def plot_barh(df,col, cmap = None, stacked=False, norm = None):
    df.plot(kind='barh', colormap=cmap, stacked=stacked)
    fig = plt.gcf()
    fig.set_size_inches(24,12)
    plt.title("Category vs {}-feedback -  cloth {}".format(col, '(Normalized)' if norm else ''), fontsize= 20)
    plt.ylabel('Category', fontsize = 18)
    plot = plt.xlabel('Frequency', fontsize=18)


# Code starts here

g_by_category = df.groupby('category').sum()
print(g_by_category)

cat_fit = df.groupby('category')['fit'].value_counts()
#print(cat_fit)

cat_fit = cat_fit.unstack()
print(cat_fit)

plot_barh(cat_fit,'fit')

# Code ends here


# --------------
# Code starts here

cat_len = df.groupby('category')['length'].value_counts()
#print(cat_len)

cat_len = cat_len.unstack()
print(cat_len)

plot_barh(cat_len,'length')
plt.show()

# Code ends here


# --------------
# Code starts here

def get_cms(x):
    if type(x) == type(1.0):
        return
    try:
        return (int(x[0])*30.48) + (int(x[4:-2])*2.54)
    except:
        return (int(x[0])*30.48)

X_train.height = X_train.height.apply(get_cms)
X_test.height = X_test.height.apply(get_cms)    
# Code ends here


# --------------
# Code starts here

print(X_train.isnull().sum())

X_train.dropna(subset=['height','length','quality'], inplace=True)
X_test.dropna(subset=['height','length','quality'], inplace=True)

y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

X_train['bra_size'].fillna(X_train['bra_size'].mean(),inplace=True)
X_test['bra_size'].fillna(X_test['bra_size'].mean(),inplace=True)

X_train['hips'].fillna(X_train['hips'].mean(),inplace=True)
X_test['hips'].fillna(X_test['hips'].mean(),inplace=True)


mode_1 = X_train['cup_size'].mode()[0]
mode_2 = X_train['cup_size'].mode()[0]

X_train['cup_size'] = X_train['cup_size'].replace(np.nan,mode_1)
X_test['cup_size'] = X_test['cup_size'].replace(np.nan,mode_2)

print(X_train.isnull().sum())
# Code ends here


# --------------
# Code starts here

X_train = pd.get_dummies(data=X_train,columns=['category','cup_size','length'], prefix=['category','cup_size','length'])
X_test = pd.get_dummies(data=X_test,columns=['category','cup_size','length'], prefix=['category','cup_size','length'])

print(X_train.columns)
# Code ends here


# --------------
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score


# Code starts here

model = DecisionTreeClassifier(random_state=6)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

score = accuracy_score(y_test,y_pred)
print(score)
precision = precision_score(y_test,y_pred,average = None) 
print(precision)

# Code ends here


# --------------
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# parameters for grid search
parameters = {'max_depth':[5,10],'criterion':['gini','entropy'],'min_samples_leaf':[0.5,1]}

# Code starts here

model = DecisionTreeClassifier(random_state = 6)

grid = GridSearchCV(estimator = model,param_grid=parameters)

grid.fit(X_train,y_train)
y_pred = grid.predict(X_test)

accuracy = grid.score(X_test,y_test)
print(accuracy)
# Code ends here


