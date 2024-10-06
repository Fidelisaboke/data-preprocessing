""" This module performs simple data preprocessing """

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Dataset
df = pd.read_csv('data.csv')

# Features (x) and target (y)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Imputing
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
x = imputer.fit_transform(x)

# Encoding
categorical_columns = [1, 3, 4, 5]

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), categorical_columns),
    ],
    remainder="passthrough"
)

x = np.array(ct.fit_transform(x))

le = LabelEncoder()
y = le.fit_transform(y)

# Split to train and test data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Standardization
sc = StandardScaler()
x_train[:,14:] = sc.fit_transform(x_train[:,14:])
x_test[:,14:] = sc.fit_transform(x_test[:,14:])

print(x_train, y_train)
print(x_test, y_test)
