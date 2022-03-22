import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_selection import SelectKBest , mutual_info_classif , f_classif, chi2
from math import sqrt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
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


X_train, X_test, y_train, y_test = train_test_split(new_X, Y, test_size = 0.20,random_state=0)

rf_clf = RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2)
rf_clf.fit(X_train, y_train)

#saving model
#with open('rf_class','wb') as f:
    #pickle.dump(rf_clf,f)

print("Accuracy score (training): {0:.3f}".format(rf_clf.score(X_train, y_train)))
print("Accuracy score (validation): {0:.3f}".format(rf_clf.score(X_test, y_test)))

#estim:100, max_depth:None, min_samples_split:2  ----> 0.976
#estim:50, max_depth:None ----> 0.976
#estim:20, max_depth:None ----> 0.974
#estim:100, max_depth:5 ----> 0.835
#estim:100, max_depth:10 ----> 0.940
#estim:100, max_depth:15 ----> 0.970
#estim:100, max_depth:None, min_samples_split:10  ----> 0.974
#estim:100, max_depth:None, min_samples_split:50  ----> 0.966

