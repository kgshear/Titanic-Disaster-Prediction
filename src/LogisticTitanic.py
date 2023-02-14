#created for kaggle Titanic ML competition

import pandas as pd
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    traindata = pd.read_csv("../data/train.csv").dropna()
    testdata = pd.read_csv("../data/test.csv")
    print(traindata)
    avg_age = traindata["Age"].mean()
    testdata["Age"] = testdata["Age"].fillna(traindata["Age"].median())
    #important features
    features = ["Pclass", "Sex", "SibSp", "Parch", "Age"]
    y_train = traindata['Survived']
    x_train = pd.get_dummies(traindata[features])
    x_test = pd.get_dummies(testdata[features])

    model = LogisticRegression()
    LogisticRegression.fit(model, x_train, y_train)
    y_pred = model.predict(x_test)

    output = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': y_pred})
    output.to_csv('../data/titanicsubmission.csv', index=False)
