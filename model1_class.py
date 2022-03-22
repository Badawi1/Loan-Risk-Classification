import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest , mutual_info_classif , f_classif, chi2
from math import sqrt
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pickle


def onehot_encode(df, column_with_perfix):
    df = df.copy()
    for column, prefix in column_with_perfix:
        dummies = pd.get_dummies(df[column], prefix=prefix)
        df = pd.concat([df, dummies], axis=1)
        df =df.drop(column, axis=1)
    return df

def preprocessing_inputs(df):
    df = df.copy()
    df = df.drop("CreditGrade", axis=1)
    df = df.drop("TotalProsperPaymentsBilled", axis=1)
    df = df.drop("ListingNumber", axis=1)
    df = df.drop("BorrowerState", axis=1)
    df = df.dropna(subset=['ProsperRating (Alpha)'])
    df['DebtToIncomeRatio'].fillna(value=df['DebtToIncomeRatio'].mean(), inplace=True)
    df['EmploymentStatusDuration'].fillna(value=df['EmploymentStatusDuration'].mean(), inplace=True)

    df["IncomeRange"].replace(
        {"$1-24,999": 12500, "$25,000-49,999": 37500, "$50,000-74,999": 62500, "$75,000-99,999": 87500,
         "$100,000+": 100000, "Not employed": 0, '$0 ': 0}, inplace=True)

    df["IsBorrowerHomeowner"] = df["IsBorrowerHomeowner"].astype(int)


    df = onehot_encode(
        df,
        column_with_perfix=[
            ('EmploymentStatus', 'E'),
            ('LoanStatus', 'L')
        ]
    )
    return df

data = pd.read_csv("LoanRiskClassification.csv")
df = preprocessing_inputs(data)
Y=df["ProsperRating (Alpha)"]
X= df.drop("ProsperRating (Alpha)", axis=1)


sel = SelectKBest(mutual_info_classif, k=15)
sel.fit(X,Y)
new_X = sel.transform(X)

#saving features
#with open('selectbest_15_class','wb') as f:
    #pickle.dump(sel,f)


X_train, X_test, y_train, y_test = train_test_split(new_X, Y, test_size = 0.20,random_state=0)

clf = DecisionTreeClassifier(criterion="entropy",min_samples_leaf=5,max_depth=None)


'''scores = cross_val_score(clf,X_train, y_train, scoring='accuracy', cv=5)
model_1_score = abs(scores.mean())
print("Decision Tree Model Cross Validation Score Is "+ str(model_1_score))'''

clf.fit(X_train,y_train)

#saving model
#with open('decision_tree_class','wb') as f:
    #pickle.dump(clf,f)

y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#gini max_depth:5 min_samples_leaf:1 = 0.8818
#gini max_depth:10 min_samples_leaf:1 = 0.9776
#gini max_depth:15 min_samples_leaf:1 = 0.9770
#entropy max_depth:5 min_samples_leaf:1 = 0.8943
#entropy max_depth:10 min_samples_leaf:1 = 0.9791
#entropy max_depth:15 min_samples_leaf:1 = 0.9797
#entropy max_depth:None min_samples_leaf:2 = 0.9776
#entropy max_depth:None min_samples_leaf:1 = 0.9787
#entropy max_depth:None min_samples_leaf:4 = 0.9796
#entropy max_depth:None min_samples_leaf:5 = 0.9805



