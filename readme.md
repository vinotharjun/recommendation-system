# Movie Recommendation API

## Overview

This project is a Flask-based API that provides movie recommendations based on user preferences. It uses a machine learning model to predict movie ratings for users, delivering personalized movie recommendations.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Install Required Packages](#install-required-packages)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
  - [Get Movie Recommendations](#get-movie-recommendations)
- [Example Request](#example-request)
- [Error Handling](#error-handling)
## Features

- Recommend top K movies for a specified user.
- Return detailed user information along with movie recommendations, including titles and genres.
- Scalable and efficient for serving multiple users.

## Technologies Used

- **Flask**: A micro web framework for Python, used for building the API.
- **PyTorch**: A deep learning library for building and training the recommendation model.
- **Pandas**: A library for data manipulation and analysis, used for handling datasets.
- **NumPy**: A library for numerical computations.
- **TQDM**: A library for creating progress bars.
- **Scikit-learn**: A library for machine learning, optionally used for preprocessing or metrics.

## Installation

### Prerequisites

Make sure you have Python 3.7 or higher installed on your system. You can check your Python version by running:

```bash
python --version
```

### clone the repo

### Install Required Packages

```
# Create a virtual environment (optional)
python -m venv venv
# Activate the virtual environment (Windows)
venv\Scripts\activate
# Activate the virtual environment (macOS/Linux)
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

```

### Running the API
start the python flask server
```
python app.py  
```

The API will be accessible at http://127.0.0.1:5000/.

### API endpoints

#### Get Movie Recommendations
```
Endpoint: /recommend/<user_id>
Method: GET
Query Parameters:
top_k (optional): The number of top recommendations to return (default is 5).

```
##### Example Request

you can use curl 
```
curl "http://127.0.0.1:5000/recommend/186?top_k=5"
```


### Response Examples
If the request is successful, you will receive a JSON response like this:

```
{
  "user_details": {
    "user_id": 186,
    "gender": "M",
    "age": 25,
    "occupation": "Engineer"
  },
  "recommendations": [
    {
      "movie_details": {
        "movie id": 1,
        "movie title": "Inception",
        "genres": "Action, Sci-Fi"
      },
      "score": 4.85
    },
    {
      "movie_details": {
        "movie id": 2,
        "movie title": "The Matrix",
        "genres": "Action, Sci-Fi"
      },
      "score": 4.80
    }
  ]
}

```

### Error Handling

If the user ID does not exist or an error occurs, the API will return a meaningful error message. For example:
```
{
  "error": "User ID not found"
}
```