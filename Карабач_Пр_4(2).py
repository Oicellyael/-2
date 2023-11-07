import pandas as pd
import numpy as np

# Завантажте дані та виведіть перші 5 рядків
data = pd.read_csv("Housing.csv")
print(data.head())

# Виберіть потрібні стовпці
columns = ['price', 'area', 'bedrooms', 'bathrooms']
data = data[columns]

# Перевірте наявність відсутніх значень
print("Кількість відсутніх значень:")
print(data.isnull().sum())

# Замініть відсутні значення середніми
data = data.fillna(data.mean())

# Нормалізація даних
data = (data - data.mean()) / data.std()

# Розділіть дані на навчальний та тестовий набори
train_ratio = 0.8
train_size = int(train_ratio * data.shape[0])
train_data = data[:train_size]
test_data = data[train_size:]

# Підготуйте дані для лінійної регресії
X_train = train_data.drop(columns=['price'])
y_train = train_data['price']
X_test = test_data.drop(columns=['price'])
y_test = test_data['price']

# Розв'яжіть нормальне рівняння аналітично
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
theta_analytical = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Виведіть коефіцієнти лінійної регресії, знайдені аналітично
print("Коефіцієнти лінійної регресії (аналітично):")
print(theta_analytical)

# Використайте бібліотеку scikit-learn для лінійної регресії
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Виведіть коефіцієнти лінійної регресії, знайдені за допомогою scikit-learn
print("Коефіцієнти лінійної регресії (за допомогою scikit-learn):")
print([model.intercept_] + model.coef_[1:].tolist())
