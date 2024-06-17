import pandas as pd        # For loading and processing the dataset
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class DataTitanic(Dataset):
    
    def __init__(self, path="./csv/train.csv", mode="training"):
        super(DataTitanic, self).__init__()
        
        self.mode = mode
        df_train = pd.read_csv(path)
        
        # We can't do anything with the Name, Ticket number, and Cabin, so we drop them.
        df_train = df_train.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1)
        
        # To make 'Sex' numeric, we replace 'female' by 0 and 'male' by 1
        df_train['Sex'] = df_train['Sex'].map({'female':0, 'male':1}).astype(int)
        
        # We replace 'Embarked' by three dummy variables 'Embarked_S', 'Embarked_C', and 'Embarked Q',
        # which are 1 if the person embarked there, and 0 otherwise.
        df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix='Embarked')], axis=1)
        df_train = df_train.drop('Embarked', axis=1)
        
        #We normalize the age and the fare by subtracting their mean and dividing by the standard deviation
        age_mean = df_train['Age'].mean()
        age_std = df_train['Age'].std()
        df_train['Age'] = (df_train['Age'] - age_mean) / age_std

        fare_mean = df_train['Fare'].mean()
        fare_std = df_train['Fare'].std()
        df_train['Fare'] = (df_train['Fare'] - fare_mean) / fare_std
        
        # A simple method to handle these missing values is to replace them by the mean age.
        df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())
        
        # Finally, we convert the Pandas dataframe to a NumPy array, and split it into a training and test set
        X_train = df_train.drop('Survived', axis=1).values.astype('float32')
        y_train = df_train['Survived'].values.astype('float32')
        
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2)
        
        X_train, y_train, X_valid, y_valid = map(torch.tensor, (X_train, y_train, X_valid, y_valid))
        
        self.X_train, self.y_train = X_train, y_train
        self.X_valid, self.y_valid = X_valid, y_valid
        
    def __getitem__(self, index):
        if self.mode == "validation":
            return (self.X_valid[index], self.y_valid[index])
        
        return (self.X_train[index], self.y_train[index])
    
    def __len__(self):
        if self.mode == "validation":
            return self.X_valid.shape[0]
        return self.X_train.shape[0]
        
    def get_training_set(self):
        return (self.X_train, self.y_train)
    
    def get_valid_set(self):
        return (self.X_valid, self.y_valid)
