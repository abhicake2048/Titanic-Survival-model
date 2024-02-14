import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
 
df = pd.read_csv("titanic.csv")

x = df.Age.mean()
df.Age.fillna(x,inplace=True)
le = LabelEncoder()
df.Sex = le.fit_transform(df.Sex)
x1 = df[["Pclass","Sex","Age","Fare"]]
y = df["Survived"]
model = tree.DecisionTreeClassifier()
X_train, X_test, y_train, y_test = train_test_split(x1,y,test_size=0.2)
model.fit(X_train,y_train)
p= model.score(X_test,y_test)
print(p)