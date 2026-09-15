from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load generated model files
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_score.pkl', 'rb'))


def get_cover_url(isbn):
    """
    Generate a book cover URL using the ISBN.
    A cover will appear when Open Library has one available.
    """
    if isbn is None:
        return ""

    isbn = str(isbn).strip()

    if isbn.endswith(".0"):
        isbn = isbn[:-2]

    return f"https://covers.openlibrary.org/b/isbn/{isbn}-M.jpg"


def find_book_cover(book_title):
    """Find ISBN for a book and generate its cover URL."""

    rows = books[books['Book-Title'] == book_title]

    if len(rows) > 0:
        isbn = rows.iloc[0]['ISBN']
        return get_cover_url(isbn)

    return ""


@app.route('/')
def index():

    book_name = list(popular_df['Book-Title'].values)
    author = list(popular_df['Book-Author'].values)
    votes = list(popular_df['num_ratings'].values)
    rating = list(popular_df['avg_rating'].values)

    image = []

    for title in book_name:
        image.append(find_book_cover(title))

    return render_template(
        'index.html',
        book_name=book_name,
        author=author,
        image=image,
        votes=votes,
        rating=rating
    )


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')


@app.route('/recommend_books', methods=['POST'])
def recommend():

    user_input = request.form.get('user_input', '').strip()

    if not user_input:
        return render_template(
            'recommend.html',
            error="Please enter a book title or author name."
        )

    search_text = user_input.lower()

    # ------------------------------------------------
    # 1. AUTHOR SEARCH
    # ------------------------------------------------

    author_matches = books[
        books['Book-Author']
        .astype(str)
        .str.lower()
        .str.contains(search_text, na=False)
    ]

    if len(author_matches) > 0:

        author_matches = author_matches.drop_duplicates(
            subset=['Book-Title']
        )

        author_matches = author_matches.head(20)

        data = []

        for _, row in author_matches.iterrows():

            title = row['Book-Title']
            author = row['Book-Author']
            isbn = row['ISBN']

            data.append([
                title,
                author,
                get_cover_url(isbn)
            ])

        return render_template(
            'recommend.html',
            data=data,
            search_type="author",
            searched=user_input
        )

    # ------------------------------------------------
    # 2. EXACT BOOK TITLE SEARCH
    # ------------------------------------------------

    exact_matches = pt.index[
        pt.index.astype(str).str.lower() == search_text
    ]

    if len(exact_matches) > 0:

        book_name = exact_matches[0]

        index = np.where(pt.index == book_name)[0][0]

        similar_items = sorted(
            list(enumerate(similarity_scores[index])),
            key=lambda x: x[1],
            reverse=True
        )[1:11]

        data = []

        for i, score in similar_items:

            title = pt.index[i]

            book_rows = books[
                books['Book-Title'] == title
            ]

            if len(book_rows) == 0:
                continue

            row = book_rows.iloc[0]

            data.append([
                title,
                row['Book-Author'],
                get_cover_url(row['ISBN'])
            ])

        return render_template(
            'recommend.html',
            data=data,
            search_type="book",
            searched=user_input
        )

    # ------------------------------------------------
    # 3. PARTIAL BOOK TITLE SEARCH
    # ------------------------------------------------

    title_matches = books[
        books['Book-Title']
        .astype(str)
        .str.lower()
        .str.contains(search_text, na=False)
    ]

    if len(title_matches) > 0:

        title_matches = title_matches.drop_duplicates(
            subset=['Book-Title']
        ).head(20)

        data = []

        for _, row in title_matches.iterrows():

            data.append([
                row['Book-Title'],
                row['Book-Author'],
                get_cover_url(row['ISBN'])
            ])

        return render_template(
            'recommend.html',
            data=data,
            search_type="title",
            searched=user_input
        )

    # ------------------------------------------------
    # 4. NOTHING FOUND
    # ------------------------------------------------

    return render_template(
        'recommend.html',
        error=f"No books or authors found for '{user_input}'.",
        searched=user_input
    )


if __name__ == '__main__':
    app.run(debug=True)