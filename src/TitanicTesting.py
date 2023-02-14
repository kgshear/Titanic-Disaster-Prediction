#created for kaggle Titanic ML competition

import pandas as pd


if __name__ == '__main__':

    x_train = pd.read_csv("../data/train.csv")

    #view dataset
    print(x_train.head())

    #determing unique values found in each (reasonable) category
    print("Unique SibSp:", pd.unique(x_train["SibSp"]))
    print("Unique Parch", pd.unique(x_train["Parch"]))
    print("Unique Age", pd.unique(x_train["Age"]))

    #determining survival statistics for different categories

    unaccompanied = x_train.loc[x_train.Parch==0]["Survived"]
    rate_unaccompanied = sum(unaccompanied)/len(unaccompanied)
    print("Percent of unaccompanied passengers who survived:", rate_unaccompanied*100)
    relatives = x_train.loc[x_train.Parch > 0]["Survived"]
    rate_relatives = sum(relatives) / len(relatives)
    print("Percent of accompanied passengers who survived:", rate_relatives * 100)


    female = x_train.loc[x_train.Sex=="female"]["Survived"]
    rate_girls = sum(female)/len(female)
    print("Percent of women passengers who survived:", rate_girls*100)
    male = x_train.loc[x_train.Sex=="male"]["Survived"]
    rate_men = sum(male)/len(male)
    print("Percent of male passengers who survived:", rate_men*100)
    first = x_train.loc[x_train.Pclass==1]["Survived"]
    rate_first = sum(first)/len(first)
    print("Percent of first class passengers who survived:", rate_first*100)
    third = x_train.loc[x_train.Pclass==3]["Survived"]
    rate_third = sum(third)/len(third)
    print("Percent of third class passengers who survived:", rate_third*100)



