from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from typing import Tuple 
import pandas as pd
import pickle

def main():
    df_features = pd.read_csv('data/preprocess/b3db.csv')
    model, report = train_model(df_features)

    with open('data/models/model.pickle', 'wb') as writer:
        writer.write(pickle.dumps(model))
    
    print(report)

def train_model(df_features:pd.DataFrame) -> Tuple[BaseEstimator, str]:
    X = df_features.drop(['label'], axis=1)
    y = df_features['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    return model, report

if __name__ == '__main__':
    main()