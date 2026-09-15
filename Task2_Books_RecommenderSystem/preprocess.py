import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("Loading datasets...")

# Load datasets
books = pd.read_csv(
    'Books.csv',
    sep=';',
    low_memory=False,
    on_bad_lines='skip',
    encoding='latin-1'
)

users = pd.read_csv(
    'Users.csv',
    sep=';',
    low_memory=False,
    on_bad_lines='skip',
    encoding='latin-1'
)

ratings = pd.read_csv(
    'Ratings.csv',
    sep=';',
    low_memory=False,
    on_bad_lines='skip',
    encoding='latin-1'
)

# Clean column names
books.columns = books.columns.str.strip()
users.columns = users.columns.str.strip()
ratings.columns = ratings.columns.str.strip()

print("Books columns:", books.columns.tolist())
print("Ratings columns:", ratings.columns.tolist())

# Rename columns according to our application
books.rename(
    columns={
        'Title': 'Book-Title',
        'Author': 'Book-Author'
    },
    inplace=True
)

ratings.rename(
    columns={
        'Rating': 'Book-Rating'
    },
    inplace=True
)

# Clean ISBN values
books['ISBN'] = (
    books['ISBN']
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.strip()
)

ratings['ISBN'] = (
    ratings['ISBN']
    .astype(str)
    .str.replace('"', '', regex=False)
    .str.strip()
)

print("Merging datasets...")

ratings_with_name = ratings.merge(
    books,
    on='ISBN'
)

print("Processing Popularity-based data...")

num_rating_df = (
    ratings_with_name
    .groupby('Book-Title')['Book-Rating']
    .count()
    .reset_index()
)

num_rating_df.rename(
    columns={'Book-Rating': 'num_ratings'},
    inplace=True
)

avg_rating_df = (
    ratings_with_name
    .groupby('Book-Title')['Book-Rating']
    .mean()
    .reset_index()
)

avg_rating_df.rename(
    columns={'Book-Rating': 'avg_rating'},
    inplace=True
)

popular_df = num_rating_df.merge(
    avg_rating_df,
    on='Book-Title'
)

popular_df = (
    popular_df[
        popular_df['num_ratings'] >= 250
    ]
    .sort_values(
        by='avg_rating',
        ascending=False
    )
    .head(50)
)

popular_df = popular_df.merge(
    books,
    on='Book-Title'
)

popular_df = popular_df.drop_duplicates(
    'Book-Title'
)

# This dataset does not contain image URLs,
# so we create an empty column for the application.
popular_df['Image-URL-M'] = ''

popular_df = popular_df[
    [
        'Book-Title',
        'Book-Author',
        'Image-URL-M',
        'num_ratings',
        'avg_rating'
    ]
]

print("Processing Collaborative Filtering...")

x = (
    ratings_with_name
    .groupby('User-ID')['Book-Rating']
    .count() > 200
)

valid_users = x[x].index

filtered_rating = ratings_with_name[
    ratings_with_name['User-ID'].isin(valid_users)
]

y = (
    filtered_rating
    .groupby('Book-Title')['Book-Rating']
    .count() >= 50
)

famous_books = y[y].index

final_ratings = filtered_rating[
    filtered_rating['Book-Title'].isin(famous_books)
]

pt = final_ratings.pivot_table(
    index='Book-Title',
    columns='User-ID',
    values='Book-Rating'
)

pt.fillna(0, inplace=True)

print("Calculating similarity scores...")

similarity_scores = cosine_similarity(pt)

print("Saving pickle files...")

pickle.dump(
    popular_df,
    open('popular.pkl', 'wb')
)

pickle.dump(
    pt,
    open('pt.pkl', 'wb')
)

pickle.dump(
    books,
    open('books.pkl', 'wb')
)

pickle.dump(
    similarity_scores,
    open('similarity_score.pkl', 'wb')
)

print("All .pkl files generated successfully!")