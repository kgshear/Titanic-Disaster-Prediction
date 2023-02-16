# Titanic-Disaster-Prediction
Program that predicts whether passengers on the Titanic survived based on their attributes

## Dataset
      Contains a training set and a test set. The ML program will be trained on the training set, and will create survival
      predictions for the test set
      
### Variables
      Survival - whether the passenger survived (0 or 1)
      
      Pclass - socio-economic class of the passenger (1st, 2nd, 3rd)
      
      Sex - gender
      
      Age - known or estimated age (may be empty)
      
      Sibsp - # of sibling/spouses aboard
      
      Parch - # of parents/children related to passenger aboard
      
      Ticket - ticket number
      
      Fare - cost of the passenger's ticket
      
      Cabin - cabin number
      
      Embarked - port where the passenger embarked (C = Cherbourg, Q = Queenstown, S = Southampton)

## TitanicTesting.py
     Turns the dataset into a dataframe and does initial exploratory analysis of the dataset. Determining the different 
     unique values in each variable, figuring out which variables resulted in a higher survival rate.
     
## LogisticTitanic.py
      Trains a logistic regression model with important features form the dataset (Pclass, Sex, SibSp, Parch, Age) and 
      generates predictions.
   
## DecisionTreeTitanic.py
      Trains a decision tree classification model with important features form the dataset (Pclass, Sex, SibSp, Parch) and 
      generates predictions.
