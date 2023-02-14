import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


if __name__ == '__main__':
    traindata = pd.read_csv("../data/train.csv")
    testdata = pd.read_csv("../data/test.csv")
    print(traindata)

    features = ["Pclass", "Sex", "SibSp", "Parch"]
    y_train = traindata['Survived']
    x_train = pd.get_dummies(traindata[features])
    x_test = pd.get_dummies(testdata[features])

    model = DecisionTreeClassifier()
    model = model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    output = pd.DataFrame({'PassengerId': testdata.PassengerId, 'Survived': y_pred})
    output.to_csv('../data/DTsubmission.csv', index=False)

