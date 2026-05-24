import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings



warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv("14-income_evaluation.csv")
print(df.head())
df.columns = df.columns.str.strip()
print("-"*40)
print("NaN Değerler")
print(df.isnull().sum())         #NAN DEĞER YOK GİBİ GÖZÜKÜYOR
print("-"*40)

categoric =  df.select_dtypes(include=["object","category"])
print(categoric)
print("-"*40)
for col in categoric.columns:
    print("SUTÜN İSMİ-->",col)
    print("Kaç Farklı Değer var --->",df[col].nunique())
    print("-" * 40)
    print("SUTÜN İSMİ-->", col)
    print(df[col].unique())   #workclass = ?, #occupation = ?
print("-"*40)
for i in ["workclass","occupation"]:
    df[df[i]==" ?"] =np.nan
    print(f"{i}=={df[i].isna().sum()}")       # Verisi ? olanları NaN değere cevirdik
print("-"*40)
from sklearn.model_selection import train_test_split
X =df.drop(columns=["income"])
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
for i in ["workclass","occupation"]:
    X_train[i] = X_train[i].fillna(X_train[i].mode()[0])
    X_test[i] = X_test[i].fillna(X_train[i].mode()[0]) #Verileri NaN değerlerlen doldurduk
y_train = y_train.fillna(y_train.mode()[0])
y_test = y_test.fillna(y_train.mode()[0]) #Verileri NaN değerlerlen doldurduk
print(X_train[["workclass","occupation"]].isna().sum())
print(X_test[["workclass","occupation"]].isna().sum()) #NaN değerler temizlenmiş Encoding işlemine Geçebiliriz
print("-"*40)

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
Le = LabelEncoder()
y_train =Le.fit_transform(y_train)
y_test =Le.transform(y_test)
print(Le.classes_)
print(np.sum(y_train==2))
print("-"*40)

cat = X_train.select_dtypes(include=["object","category"])
print(cat.columns[0:8])


encoder = ColumnTransformer(transformers=[("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),cat.columns[0:8])],
                            remainder="passthrough")

X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
print(X_train)

print("-"*40)
print(type(X_train))
print(X_test.shape)
print("-"*40)
from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()
DTC.fit(X_train,y_train)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
pred = DTC.predict(X_test)
print("-"*40)
print(confusion_matrix(y_test,pred))
print("-"*40)
print(classification_report(y_test,pred))
print("-"*40)
print(accuracy_score(y_test,pred))
print("-"*40)

from sklearn.model_selection import RandomizedSearchCV
dctparams = {"criterion":["gini", "entropy", "log_loss"],
             "max_depth":[15,100,150,200],
             "max_features":["sqrt","log2",35],
             "min_samples_leaf":[1,2,3,4]}
random = RandomizedSearchCV(DTC,dctparams,n_iter=10,random_state=42,verbose=1,n_jobs=-1,cv=10)
random.fit(X_train,y_train)
pred2 = random.predict(X_test)
print("-"*40)
print(confusion_matrix(y_test,pred2))
print("-"*40)
print(classification_report(y_test,pred2))
print("-"*40)
print(accuracy_score(y_test,pred2))
print("-"*40)
print(random.best_estimator_) # {'min_samples_leaf': 1, 'max_features': 35, 'max_depth': 15, 'criterion': 'entropy'}

from sklearn.tree import plot_tree

best = random.best_estimator_
features = encoder.get_feature_names_out()
plt.figure(figsize=(10,10))

plot_tree(
    best,
    max_depth=3,

    feature_names=features,
    class_names=[str(cls) for cls in Le.classes_],
    filled=True,
    rounded=True,
    fontsize=10
)

plt.show()