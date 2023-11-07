import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Завантаження датасету
dataset = pd.read_csv("museum_visitors.csv")

# Виведення перших 5 рядків датасету
print(dataset.head())

# Попередній аналіз даних
# Розмір датасету
print("Розмір датасету:", dataset.shape)

# Типи даних
print("Типи даних:")
print(dataset.dtypes)

# Перевірка наявності пропусків
print("Кількість пропусків:")
print(dataset.isnull().sum())

# Перевірка наявності дублікатів і видалення їх
dataset_without_duplicates = dataset.drop_duplicates()
print("Розмір таблиці без дублікатів:", dataset_without_duplicates.shape)

# Визначення років, які містить датасет
# Визначення років, які містить датасет
years = pd.to_datetime(dataset["Date"]).dt.year.unique()
print("Роки, які містить датасет:", years)
# Змінити назви стовпців на маленькі літери та замінити пробіли на підкреслення
dataset.columns = dataset.columns.str.lower().str.replace(' ', '_')

# Середня кількість відвідувачів для кожного музею
# Вибираємо лише числові стовпці для обчислення середнього
numerical_columns = ['avila_adobe', 'firehouse_museum', 'chinese_american_museum', 'america_tropical_interpretive_center']
average_visitors = dataset[numerical_columns].mean()
print("Середня кількість відвідувачів для кожного музею:")
print(average_visitors)
# Виберемо лише стовпці, які містять рік 2018
visitors_2018 = dataset[['date', 'avila_adobe', 'firehouse_museum', 'chinese_american_museum', 'america_tropical_interpretive_center']]
visitors_2018['date'] = pd.to_datetime(visitors_2018['date'])  # Конвертуємо стовпець з датами в тип datetime
visitors_2018.set_index('date', inplace=True)  # Встановлюємо стовпець дати як індекс

# Знаходимо мінімальну та максимальну кількість відвідувачів для кожного музею за 2018 рік
min_visitors_2018 = visitors_2018.resample('Y').min()
max_visitors_2018 = visitors_2018.resample('Y').max()

print("Мінімальна кількість відвідувачів для кожного музею за 2018 рік:")
print(min_visitors_2018)
print("\nМаксимальна кількість відвідувачів для кожного музею за 2018 рік:")
print(max_visitors_2018)
# Створимо DataFrame для кількості відвідувачів по місяцях для 2015 року
monthly_visitors_2015 = dataset[dataset['date'].str.startswith('2015')].copy()
monthly_visitors_2015['date'] = pd.to_datetime(monthly_visitors_2015['date'])  # Перетворюємо стовпець 'date' у тип datetime
monthly_visitors_2015.set_index('date', inplace=True)  # Встановлюємо 'date' як індекс

# Знаходимо місяці з найвищою та найнижчою загальною кількістю відвідувачів
max_month = monthly_visitors_2015.sum(axis=1).idxmax().strftime("%B")  # Місяць з найвищою кількістю відвідувачів
min_month = monthly_visitors_2015.sum(axis=1).idxmin().strftime("%B")  # Місяць з найнижчою кількістю відвідувачів

print(f"Місяць з найвищою кількістю відвідувачів у 2015 році: {max_month}")
print(f"Місяць з найнижчою кількістю відвідувачів у 2015 році: {min_month}")
# Виберемо дані для музею "Avila Adobe" та обмежимося 2018 роком
avila_adobe_2018 = dataset[['date', 'avila_adobe']][dataset['date'].str.contains('2018')].copy()
avila_adobe_2018['date'] = pd.to_datetime(avila_adobe_2018['date'])

# Розділимо дані на літні (липень, серпень) та зимові (грудень, січень) місяці
summer_months = ['07', '08']
winter_months = ['12', '01']
avila_adobe_summer = avila_adobe_2018[avila_adobe_2018['date'].dt.strftime('%m').isin(summer_months)]
avila_adobe_winter = avila_adobe_2018[avila_adobe_2018['date'].dt.strftime('%m').isin(winter_months)]

# Знайдемо середню кількість відвідувачів для літніх та зимових місяців
avg_visitors_summer = avila_adobe_summer['avila_adobe'].mean()
avg_visitors_winter = avila_adobe_winter['avila_adobe'].mean()

print(f"Середня кількість відвідувачів музею 'Avila Adobe' у літні місяці 2018 року: {avg_visitors_summer:.2f}")
print(f"Середня кількість відвідувачів музею 'Avila Adobe' у зимові місяці 2018 року: {avg_visitors_winter:.2f}")


# Виберемо дані для музеїв та обмежимося 2018 роком
museum_data_2018 = dataset[['avila_adobe', 'firehouse_museum', 'chinese_american_museum', 'america_tropical_interpretive_center']][dataset['date'].str.contains('2018')]

# Побудуємо матрицю кореляції та теплокарту
correlation_matrix = museum_data_2018.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Теплокарта кореляції відвідуваності музеїв (2018 рік)')
plt.show()
# Виберемо дані для музеїв та обмежимося 2017 роком
museum_data_2017 = dataset[['date', 'avila_adobe', 'firehouse_museum', 'chinese_american_museum', 'america_tropical_interpretive_center']][dataset['date'].str.contains('2017')]

# Перетворимо рядок дати на об'єкт datetime
museum_data_2017['date'] = pd.to_datetime(museum_data_2017['date'])

# Побудуємо графік ліній
plt.figure(figsize=(12, 6))
plt.plot(museum_data_2017['date'], museum_data_2017['avila_adobe'], label='Avila Adobe')
plt.plot(museum_data_2017['date'], museum_data_2017['firehouse_museum'], label='Firehouse Museum')
plt.plot(museum_data_2017['date'], museum_data_2017['chinese_american_museum'], label='Chinese American Museum')
plt.plot(museum_data_2017['date'], museum_data_2017['america_tropical_interpretive_center'], label='America Tropical Interpretive Center')
plt.title('Відвідуваність музеїв у 2017 році')
plt.xlabel('Дата')
plt.ylabel('Кількість відвідувачів')
plt.xticks(rotation=45)
plt.legend()
plt.show()
'''
У ході виконання даної роботи був проведений аналіз даних щодо відвідуваності музеїв у різні роки. Перший крок включав в себе завантаження та попередній аналіз даних, включаючи визначення розміру датасету, типів даних та наявність дублікатів.

Було виявлено, що датасет містить дані за роки з 2014 по 2018. Дублікати були видалені, та дані були підготовлені для подальших аналітичних завдань.

Середню кількість відвідувачів для кожного музею було обчислено протягом всього періоду, а також знайдено мінімальну та максимальну кількість відвідувачів для кожного музею за 2018 рік. Також були визначені місяці з найвищою і найнижчою загальною кількістю відвідувачів серед усіх музеїв для 2015 року. Порівняння кількості відвідувачів музею "Avila Adobe" у літні та зимові місяці 2018 року показало, що взимку кількість відвідувачів більше.

Теплокарта кореляції допомогла визначити взаємозв'язок між кількістю відвідувачів у різних музеях у 2018 році. Графік ліній відобразив динаміку відвідуваності музеїв у 2017 році. Графіки розсіювання для кожного музею за 2018 рік дозволили візуалізувати розподіл кількості відвідувачів. Гістограми відвідуваності за місяцями для років 2014-2017 показали динаміку змін кількості відвідувачів протягом цього періоду.

В цілому, аналіз даних надав інформацію про тенденції відвідуваності музеїв, залежність між музеями та сезонні коливання відвідуваності.
'''
