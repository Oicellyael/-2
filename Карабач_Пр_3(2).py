import pandas as pd

# Зчитуємо дані з CSV файлу
file_path = "C:/Users/andrey/PycharmProjects/pythonProject/bestsellers with categories.csv"
data = pd.read_csv(file_path)

# Виводимо перші 10 рядків
print(data.head(10))

# Виводимо розміри датасету
print("Розмір датасету:", data.shape)
# Змінюємо назви стовпців
data.columns = ['name', 'author', 'user_rating', 'reviews', 'price', 'year', 'genre']
# Визначаємо унікальні жанри в стовпці "genre"
unique_genres = data['genre'].unique()
print("Унікальні жанри:", unique_genres)
# Обчислюємо максимальну, мінімальну, медіанну і середню ціну
max_price = data['price'].max()
min_price = data['price'].min()
median_price = data['price'].median()
mean_price = data['price'].mean()

print("Максимальна ціна:", max_price)
print("Мінімальна ціна:", min_price)
print("Медіанна ціна:", median_price)
print("Середня ціна:", mean_price)
max_rating = data['user_rating'].max()
print("Найвищий рейтинг:", max_rating)
highest_rated_books = data[data['user_rating'] == max_rating]
number_of_highest_rated_books = highest_rated_books.shape[0]
print("Кількість книг з найвищим рейтингом:", number_of_highest_rated_books)
most_reviewed_book_index = data['reviews'].idxmax()
most_reviewed_book = data.loc[most_reviewed_book_index]
print("Книга з найбільшою кількістю відгуків:", most_reviewed_book['name'])
books_2010 = data[data['year'] == 2010]
most_expensive_2010_book_index = books_2010['price'].idxmax()
most_expensive_2010_book = books_2010.loc[most_expensive_2010_book_index]
print("Найдорожча книга серед книг, що потрапили до Топ-50 у 2010 році:", most_expensive_2010_book['name'])
fiction_books_2012 = data[(data['genre'] == 'Fiction') & (data['year'] == 2012)]
number_of_fiction_books_2012 = fiction_books_2012.shape[0]
print("Кількість книг жанру Fiction, що потрапили до Топ-50 у 2012 році:", number_of_fiction_books_2012)


genre_prices = data.groupby('genre').agg({'price': ['max', 'min']})
print("Максимальна і мінімальна ціни для жанру Fiction і NonFiction:")
print(genre_prices)
# Фільтруємо дані для жанрів "Fiction" і "NonFiction"
fiction_nonfiction_data = data[data['genre'].isin(['Fiction', 'NonFiction'])]

# Групуємо дані за жанр і обчислюємо максимальну і мінімальну ціну
result = fiction_nonfiction_data.groupby('genre')['price'].agg(['max', 'min'])

# Виводимо результат
print(result)
