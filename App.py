# import streamlit as st
# import pandas as pd
# from surprise import Dataset, Reader, SVD

# # Load the datasets
# @st.cache_data
# def load_data():
#     genome_scores = pd.read_csv('/Users/da-m1-18/Downloads/genome_scores.csv')
#     genome_tags = pd.read_csv('/Users/da-m1-18/Downloads/genome_tags.csv')
#     imdb_data = pd.read_csv('/Users/da-m1-18/Downloads/imdb_data.csv')
#     links = pd.read_csv('/Users/da-m1-18/Downloads/links.csv')
#     movies = pd.read_csv('/Users/da-m1-18/Downloads/movies.csv')
#     sample_submission = pd.read_csv('/Users/da-m1-18/Downloads/sample_submission.csv')
#     tags = pd.read_csv('/Users/da-m1-18/Downloads/tags.csv')
#     test = pd.read_csv('/Users/da-m1-18/Downloads/test.csv')
#     train = pd.read_csv('/Users/da-m1-18/Downloads/train (1).csv')
#     return genome_scores, genome_tags, imdb_data, links, movies, sample_submission, tags, test, train

# # Load data
# genome_scores, genome_tags, imdb_data, links, movies, sample_submission, tags, test, train = load_data()

# # Merge datasets (example: merging train with movies to get movie titles)
# train = pd.merge(train, movies, on='movieId', how='left')
# test = pd.merge(test, movies, on='movieId', how='left')

# # Prepare the data for the Surprise library
# @st.cache_resource
# def train_model():
#     reader = Reader(rating_scale=(0.5, 5.0))
#     train_data = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)
#     trainset = train_data.build_full_trainset()
#     algo = SVD()
#     algo.fit(trainset)
#     return algo

# # Train the model
# algo = train_model()

# # Streamlit app
# st.title("Movie Recommendation System")
# st.write("Enter your user ID to get movie recommendations:")

# user_id = st.number_input("User ID", min_value=1, step=1)

# if st.button("Recommend"):
#     with st.spinner('Generating recommendations...'):
#         # Get all movie IDs
#         all_movie_ids = movies['movieId'].unique()

#         # Predict ratings for all movies for the given user
#         predictions = [algo.predict(user_id, movie_id) for movie_id in all_movie_ids]

#         # Sort predictions by estimated rating
#         top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

#         # Get top 10 movie IDs
#         top_movie_ids = [pred.iid for pred in top_predictions[:10]]

#         # Get movie titles
#         recommended_movies = movies[movies['movieId'].isin(top_movie_ids)]

#         st.write("Top 10 movie recommendations for you:")
#         for _, row in recommended_movies.iterrows():
#             st.write(f"{row['title']}")

# if st.button("Show All Dataframes"):
#     st.write("Genome Scores", genome_scores.head())
#     st.write("Genome Tags", genome_tags.head())
#     st.write("IMDB Data", imdb_data.head())
#     st.write("Links", links.head())
#     st.write("Movies", movies.head())
#     st.write("Sample Submission", sample_submission.head())
#     st.write("Tags", tags.head())
#     st.write("Test", test.head())
#     st.write("Train", train.head())


# import streamlit as st
# import pandas as pd
# from surprise import Dataset, Reader, SVD

# # Set page configuration
# st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")

# # Load the datasets with encoding handling
# @st.cache_data
# def load_data():
#     encodings = ['utf-8', 'ISO-8859-1', 'latin1']
#     files = [
#         '/Users/da-m1-18/Downloads/genome_scores.csv',
#         '/Users/da-m1-18/Downloads/genome_tags.csv',
#         '/Users/da-m1-18/Downloads/imdb_data.csv',
#         '/Users/da-m1-18/Downloads/links.csv',
#         '/Users/da-m1-18/Downloads/movies.csv',
#         '/Users/da-m1-18/Downloads/sample_submission.csv',
#         '/Users/da-m1-18/Downloads/tags.csv',
#         '/Users/da-m1-18/Downloads/test.csv',
#         '/Users/da-m1-18/Downloads/train (1).csv',
#         '/Users/da-m1-18/Downloads/MovieGenre.csv'
#     ]
#     dataframes = []
    
#     for file in files:
#         for encoding in encodings:
#             try:
#                 df = pd.read_csv(file, encoding=encoding)
#                 dataframes.append(df)
#                 break
#             except UnicodeDecodeError:
#                 continue
#         else:
#             raise ValueError(f"Could not decode {file} with provided encodings.")
    
#     return dataframes

# # Load data
# (genome_scores, genome_tags, imdb_data, links, movies, sample_submission, tags, test, train, movie_posters) = load_data()

# # Merge datasets (example: merging train with movies to get movie titles)
# train = pd.merge(train, movies, on='movieId', how='left')
# test = pd.merge(test, movies, on='movieId', how='left')
# links = pd.merge(links, movies, on='movieId', how='left')
# links = pd.merge(links, movie_posters[['imdbId', 'Poster']], left_on='imdbId', right_on='imdbId', how='left')

# # Default image URL for movies without cover wallpapers
# default_image_url = "https://static.vecteezy.com/system/resources/previews/005/903/347/non_2x/gold-abstract-letter-s-logo-for-negative-video-recording-film-production-free-vector.jpg"

# # Prepare the data for the Surprise library
# @st.cache_resource
# def train_model():
#     reader = Reader(rating_scale=(0.5, 5.0))
#     train_data = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)
#     trainset = train_data.build_full_trainset()
#     algo = SVD()
#     algo.fit(trainset)
#     return algo

# # Train the model
# algo = train_model()

# # Custom CSS for background and styles
# st.markdown(
#     """
#     <style>
#     .main {
#         background: url('https://st3.depositphotos.com/1064045/15061/i/450/depositphotos_150614902-stock-photo-unusual-cinema-concept-3d-illustration.jpg');
#         background-size: cover;
#     }
#     .title {
#         color: #ffffff;
#         font-size: 3em;
#     }
#     .recommendation-box {
#         background: rgba(255, 255, 255, 0.8);
#         padding: 20px;
#         border-radius: 10px;
#         display: grid;
#         grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
#         gap: 20px;
#     }
#     .movie {
#         text-align: center;
#     }
#     .movie img {
#         max-width: 150px;
#         border-radius: 10px;
#     }
#     .movie-title {
#         color: #000000;
#         font-weight: bold;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# # Streamlit app
# st.markdown('<h1 class="title">🎬 Movie Recommendation System 🎬</h1>', unsafe_allow_html=True)
# st.write("Enter your details to get movie recommendations:")

# # User input
# user_id = st.number_input("User ID", min_value=1, step=1)
# rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5)
# genre = st.text_input("Genre")
# title = st.text_input("Movie Title")
# year = st.text_input("Year")

# if st.button("Recommend"):
#     with st.spinner('Generating recommendations...'):
#         # Filter movies based on user input
#         filtered_movies = movies
#         if genre:
#             filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(genre, case=False, na=False)]
#         if title:
#             filtered_movies = filtered_movies[filtered_movies['title'].str.contains(title, case=False, na=False)]
#         if year:
#             filtered_movies = filtered_movies[filtered_movies['title'].str.contains(f"({year})", case=False, na=False)]
        
#         # Get all filtered movie IDs
#         filtered_movie_ids = filtered_movies['movieId'].unique()

#         # Predict ratings for all filtered movies for the given user
#         predictions = [algo.predict(user_id, movie_id) for movie_id in filtered_movie_ids]

#         # Sort predictions by estimated rating
#         top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

#         # Filter by rating
#         top_predictions = [pred for pred in top_predictions if pred.est >= rating]

#         # Get top 10 movie IDs
#         top_movie_ids = [pred.iid for pred in top_predictions[:10]]

#         # Get movie details
#         recommended_movies = links[links['movieId'].isin(top_movie_ids)]

#         if recommended_movies.empty:
#             st.write("No recommendations found based on your criteria.")
#         else:
#             st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
#             st.write("Top 10 movie recommendations for you:")
#             for _, row in recommended_movies.iterrows():
#                 poster = row['Poster'] if pd.notna(row['Poster']) else default_image_url
#                 st.markdown(f"""
#                     <div class="movie">
#                         <img src="{poster}" alt="{row['title']}">
#                         <div class="movie-title">{row['title']}</div>
#                     </div>
#                 """, unsafe_allow_html=True)
#             st.markdown('</div>', unsafe_allow_html=True)

# if st.button("Show All Dataframes"):
#     st.write("Genome Scores", genome_scores.head())
#     st.write("Genome Tags", genome_tags.head())
#     st.write("IMDB Data", imdb_data.head())
#     st.write("Links", links.head())
#     st.write("Movies", movies.head())
#     st.write("Sample Submission", sample_submission.head())
#     st.write("Tags", tags.head())
#     st.write("Test", test.head())
#     st.write("Train", train.head())
#     st.write("Movie Posters", movie_posters.head())


import streamlit as st
import pandas as pd
from surprise import Dataset, Reader, SVD

# Set page configuration
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")

# Load the datasets with encoding handling
@st.cache_data
def load_data():
    encodings = ['utf-8', 'ISO-8859-1', 'latin1']
    files = [
        '/Users/da-m1-18/Downloads/genome_scores.csv',
        '/Users/da-m1-18/Downloads/genome_tags.csv',
        '/Users/da-m1-18/Downloads/imdb_data.csv',
        '/Users/da-m1-18/Downloads/links.csv',
        '/Users/da-m1-18/Downloads/movies.csv',
        '/Users/da-m1-18/Downloads/sample_submission.csv',
        '/Users/da-m1-18/Downloads/tags.csv',
        '/Users/da-m1-18/Downloads/test.csv',
        '/Users/da-m1-18/Downloads/train (1).csv',
        '/Users/da-m1-18/Downloads/MovieGenre.csv'
    ]
    dataframes = []
    
    for file in files:
        for encoding in encodings:
            try:
                df = pd.read_csv(file, encoding=encoding)
                dataframes.append(df)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError(f"Could not decode {file} with provided encodings.")
    
    return dataframes

# Load data
(genome_scores, genome_tags, imdb_data, links, movies, sample_submission, tags, test, train, movie_posters) = load_data()

# Merge datasets (example: merging train with movies to get movie titles)
train = pd.merge(train, movies, on='movieId', how='left')
test = pd.merge(test, movies, on='movieId', how='left')
links = pd.merge(links, movies, on='movieId', how='left')
links = pd.merge(links, movie_posters[['imdbId', 'Poster']], left_on='imdbId', right_on='imdbId', how='left')

# Default image URL for movies without cover wallpapers
default_image_url = "https://static.vecteezy.com/system/resources/previews/005/903/347/non_2x/gold-abstract-letter-s-logo-for-negative-video-recording-film-production-free-vector.jpg"

# Prepare the data for the Surprise library
@st.cache_resource
def train_model():
    reader = Reader(rating_scale=(0.5, 5.0))
    train_data = Dataset.load_from_df(train[['userId', 'movieId', 'rating']], reader)
    trainset = train_data.build_full_trainset()
    algo = SVD()
    algo.fit(trainset)
    return algo

# Train the model
algo = train_model()

# Custom CSS for background and styles
st.markdown(
    """
    <style>
    .main {
        background: url('https://st3.depositphotos.com/1064045/15061/i/450/depositphotos_150614902-stock-photo-unusual-cinema-concept-3d-illustration.jpg');
        background-size: cover;
    }
    .title {
        color: #ffffff;
        font-size: 3em;
    }
    .recommendation-box {
        background: rgba(255, 255, 255, 0.8);
        padding: 20px;
        border-radius: 10px;
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
        gap: 20px;
    }
    .movie {
        text-align: center;
    }
    .movie img {
        max-width: 150px;
        border-radius: 10px;
    }
    .movie-title {
        color: #000000;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.markdown('<h1 class="title">🎬 Movie Recommendation System 🎬</h1>', unsafe_allow_html=True)
st.write("Enter your details to get movie recommendations:")

# User input
rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5)
all_genres = sorted(set([genre for sublist in movies['genres'].str.split('|') for genre in sublist if isinstance(sublist, list)]))
genre = st.selectbox("Genre", ["Any"] + all_genres)
title = st.text_input("Movie Title")
year = st.text_input("Year")
director = st.text_input("Director")

if st.button("Recommend"):
    with st.spinner('Generating recommendations...'):
        # Filter movies based on user input
        filtered_movies = movies
        if genre and genre != "Any":
            filtered_movies = filtered_movies[filtered_movies['genres'].str.contains(genre, case=False, na=False)]
        if title:
            filtered_movies = filtered_movies[filtered_movies['title'].str.contains(title, case=False, na=False)]
        if year:
            filtered_movies = filtered_movies[filtered_movies['title'].str.contains(f"({year})", case=False, na=False)]
        if director:
            filtered_movies = filtered_movies[filtered_movies['movieId'].isin(imdb_data[imdb_data['director'].str.contains(director, case=False, na=False)]['movieId'])]

        # Get all filtered movie IDs
        filtered_movie_ids = filtered_movies['movieId'].unique()

        # Predict ratings for all filtered movies
        predictions = [algo.predict(0, movie_id) for movie_id in filtered_movie_ids]  # user id 0 to get general predictions

        # Sort predictions by estimated rating
        top_predictions = sorted(predictions, key=lambda x: x.est, reverse=True)

        # Filter by rating
        top_predictions = [pred for pred in top_predictions if pred.est >= rating]

        # Get top 10 movie IDs
        top_movie_ids = [pred.iid for pred in top_predictions[:10]]

        # Get movie details
        recommended_movies = links[links['movieId'].isin(top_movie_ids)]

        if recommended_movies.empty:
            st.write("No recommendations found based on your criteria.")
        else:
            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
            st.write("Top 10 movie recommendations for you:")
            for _, row in recommended_movies.iterrows():
                poster = row['Poster'] if pd.notna(row['Poster']) else default_image_url
                st.markdown(f"""
                    <div class="movie">
                        <img src="{poster}" alt="{row['title']}">
                        <div class="movie-title">{row['title']}</div>
                    </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

if st.button("Show All Dataframes"):
    st.write("Genome Scores", genome_scores.head())
    st.write("Genome Tags", genome_tags.head())
    st.write("IMDB Data", imdb_data.head())
    st.write("Links", links.head())
    st.write("Movies", movies.head())
    st.write("Sample Submission", sample_submission.head())
    st.write("Tags", tags.head())
    st.write("Test", test.head())
    st.write("Train", train.head())
    st.write("Movie Posters", movie_posters.head())

