from flask import Flask, request, jsonify
import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset

class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):
        users = self.users[item]
        movies = self.movies[item]
        ratings = self.ratings[item]
        return {
            "users": torch.tensor(users, dtype=torch.long),
            "movies": torch.tensor(movies, dtype=torch.long),
            "ratings": torch.tensor(ratings, dtype=torch.float),
        }


import torch
from torch import nn

class RatingScaler(nn.Module):
  def __init__(self):
    super(RatingScaler, self).__init__()

  def forward(self, x):
    return torch.clamp(x, min=0.0, max=5.0)

class RecommendationSystemModel(nn.Module):
    def __init__(
        self,
        num_users,
        num_movies,
        embedding_size=256,
        hidden_dim=256,
        dropout_rate=0.2,
    ):
        super(RecommendationSystemModel, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.user_embedding = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.embedding_size
        )
        self.movie_embedding = nn.Embedding(
            num_embeddings=self.num_movies, embedding_dim=self.embedding_size
        )
        self.fc1 = nn.Linear(2 * self.embedding_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.rating_scaler = RatingScaler()

    def forward(self, users, movies):
        user_embedded = self.user_embedding(users)
        movie_embedded = self.movie_embedding(movies)
        combined = torch.cat([user_embedded, movie_embedded], dim=1)

        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)
        output = self.rating_scaler(output)

        return output
    def recommend_top_5(self, user_id, all_movie_ids, device='cpu'):
        user_tensor = torch.tensor([user_id] * len(all_movie_ids), dtype=torch.long).to(device)
        movie_tensor = torch.tensor(all_movie_ids, dtype=torch.long).to(device)

        self.eval()
        with torch.no_grad():
            predicted_ratings = self.forward(user_tensor, movie_tensor).squeeze(1)

        top_5_ratings, top_5_movie_indices = torch.topk(predicted_ratings, k=5)
        top_5_movie_ids = [all_movie_ids[i] for i in top_5_movie_indices.cpu().numpy()]
        return top_5_movie_ids

df_users = pd.read_csv("processed_data/users_data.csv")
df_items = pd.read_csv("processed_data/movies_data.csv")

def combine_genres(df):
    genre_columns = df.columns.difference(['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL'])
    def get_genre_list(row):
        genres = [col for col in genre_columns if row[col] == 1]
        return ', '.join(genres) if genres else 'Unknown'
    df['genres'] = df.apply(get_genre_list, axis=1)
    df = df.drop(columns=genre_columns)
    return df

df_items = combine_genres(df_items)

num_users = df_users['user_id'].nunique()
num_movies = df_items['movie id'].nunique()
model = RecommendationSystemModel(num_users=num_users, num_movies=num_movies)

model.load_state_dict(torch.load("models/recommendation_model.pth"))

app = Flask(__name__)


def recommend_movies_for_user_with_details(model, user_id, df_users, df_movies, top_k=5, device='cpu'):
    """
    Recommend top K movies for a given user ID with user and movie details.
    """
    model.to(device)
    model.eval()

    all_movie_ids = df_movies['movie id'].unique()
    all_movie_ids = all_movie_ids - 1
    user_tensor = torch.tensor([user_id] * len(all_movie_ids), dtype=torch.long).to(device)
    movie_tensor = torch.tensor(all_movie_ids, dtype=torch.long).to(device)

    with torch.no_grad():
        predictions = model(user_tensor, movie_tensor).squeeze(1)

    predictions = predictions.cpu().numpy()
    movie_predictions = list(zip(all_movie_ids, predictions))
    movie_predictions.sort(key=lambda x: x[1], reverse=True)

    top_movies = movie_predictions[:top_k]
    top_movie_ids = [movie_id for movie_id, _ in top_movies]
    top_movie_scores = [float(score) for _, score in top_movies]  # Convert to float

    user_details = df_users[df_users['user_id'] == user_id + 1]
    if user_details.empty:
        raise ValueError("User not found")  # Raise error if user is not found

    user_details = user_details.to_dict('records')[0]
    recommended_movies = []
    for movie_id, score in zip(top_movie_ids, top_movie_scores):
        movie_details = df_movies[df_movies['movie id'] == movie_id + 1].to_dict('records')
        if movie_details:
            recommended_movies.append({
                'movie_details': movie_details[0],
                'score': score 
            })

    return {
        'user_details': user_details,
        'recommendations': recommended_movies
    }

@app.route('/recommend/<int:user_id>', methods=['GET'])
def get_movie_recommendations(user_id):
    try:
        top_k = request.args.get('top_k', default=5, type=int)
        adjusted_user_id = user_id - 1
        recommendations = recommend_movies_for_user_with_details(model, adjusted_user_id, df_users, df_items, top_k=top_k)

        response = {
            "statusCode": 200,
            'user_details': recommendations['user_details'],
            'recommendations': recommendations['recommendations']
        }
        return jsonify(response)

    except ValueError as e:
        return jsonify({"statusCode": 404, "error": str(e)}), 404  # User not found

    except Exception as e:
        return jsonify({"statusCode": 500, "error": "Internal Server Error", "message": str(e)}), 500  # Runtime error

if __name__ == '__main__':
    app.run(debug=True)