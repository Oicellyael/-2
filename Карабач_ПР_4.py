import pandas as pd

data = pd.read_csv("C:/Users/andrey/PycharmProjects/pythonProject/Housing.csv")
print(data.head())
columns = ['price', 'area', 'bedrooms', 'bathrooms']
selected_data = data[columns]
# Перевірка типів даних і виявлення пропусків
print(selected_data.info())

# Заміна "-" на NaN
selected_data = selected_data.replace('-', pd.NA)

# Зміна типів даних на float
selected_data = selected_data.astype('float')

# Заповнення пропусків середніми значеннями
selected_data = selected_data.fillna(selected_data.mean())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
normalized_data = scaler.fit_transform(selected_data)
#Розділити дані на навчальний та тестовий набори.
from sklearn.model_selection import train_test_split

X = normalized_data[:, 1:]
y = normalized_data[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Побудувати модель лінійної регресії та навчити її на навчальних даних.
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
#Оцінити якість моделі на тестових даних, використовуючи RMSE та R^2.
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R^2:", r2)

#Застосувати крос-валідацію для підвищення якості моделі та порівняти результати.
from sklearn.model_selection import cross_val_score

cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_rmse = (cross_val_scores * -1) ** 0.5

print("Cross-validated RMSE:", cross_val_rmse)
#Вивести прогнозовані ціни на будинки для тестового набору даних.
y_pred_test = model.predict(X_test)
predicted_prices = scaler.inverse_transform(
    pd.concat([pd.Series(y_pred_test), pd.DataFrame(X_test, columns=columns[1:])], axis=1)
)
print(predicted_prices)


#


