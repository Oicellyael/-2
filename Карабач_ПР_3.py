import pandas as pd

# Прочитайте CSV файл з вказанням кодування
data = pd.read_csv("C:/Users/andrey/PycharmProjects/pythonProject/Global YouTube Statistics.csv", encoding="windows-1254")
# Виведіть перші п'ять рядків
print(data.head())

# Виведіть розміри датасету
print("Розміри датасету:", data.shape)

# Перевірте пропуски та замініть їх на NaN, а потім змініть тип даних на числовий
missing_values = data.isna().sum()
print("Кількість пропусків у кожному стовпці:")
print(missing_values)

data.fillna(value=float('nan'), inplace=True)
data = data.apply(pd.to_numeric, errors='coerce')
# Перевірте кількість пропусків ще раз
missing_values = data.isna().sum()
print("Кількість пропусків після заміни на середні значення:")
print(missing_values)

# Перевірте унікальні країни у колонці "Country"
unique_countries = data['Country'].nunique()
print("Кількість унікальних країн:", unique_countries)
# Побудуйте діаграму розподілу переглядів


# Визначте максимальну, мінімальну та середню кількість переглядів
max_views = data['video views'].max()
min_views = data['video views'].min()
mean_views = data['video views'].mean()
print("Максимальна кількість переглядів:", max_views)
print("Мінімальна кількість переглядів:", min_views)
print("Середня кількість переглядів:", mean_views)

# Знайдіть країн
data['uploads'].fillna(0, inplace=True)
country_with_most_uploads = data[data['uploads'] == data['uploads'].max()]['Country'].values[0]
print("Країна з найбільшою кількістю завантажень на YouTube:", country_with_most_uploads)

