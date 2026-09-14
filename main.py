import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


df = pd.read_csv("Social_Network_Ads.csv")

X = df[["Age", "EstimatedSalary"]]

y = df["Purchased"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression()

model.fit(X_train, y_train)


print("Training data:", X_train.shape)

print("Testing data:", X_test.shape)

y_pred = model.predict(X_test)

print(y_pred)


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)












































# df = pd.read_csv("Social_Network_Ads.csv")

# print(df.head())
# print(df.shape)
# print(df.isnull().sum())


# df = pd.read_csv("Social_Network_Ads.csv")

# X = df[["Age", "EstimatedSalary"]]
# y = df["Purchased"]

# print(X.head())
# print(y.head())
