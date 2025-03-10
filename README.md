# Movie Recommendation System
## Overview

The Movie Recommendation System is an item-based filtering recommendation engine built using Python. It leverages collaborative filtering techniques to suggest similar movies based on user ratings.

## Features

- Uses item-based filtering for efficient and scalable recommendations.
- Utilizes cosine similarity for finding similar movies.
- Implements data filtering to remove sparse data and enhance accuracy.
- Uses scipy's csr_matrix for memory-efficient sparse matrix representation.
- Built using scikit-learn's Nearest Neighbors algorithm.
  
## Dependencies

```python
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

## Data Preprocessing

- **Loading datasets:** The MovieLens dataset (CSV files) is loaded using Pandas.
- **Data Cleaning:**
    - Filtering movies with at least 10 ratings.
    - Filtering users who have rated at least 50 movies.
- **Sparse Matrix Conversion:**
    - Converts the dataset into a pivot table with movies as rows and users as columns.
    - Uses csr_matrix from scipy.sparse for memory efficiency.
      
## Model Training

- Implements **K-Nearest Neighbors (KNN)** with cosine similarity.
- Trains the model using the sparse matrix representation of movie ratings.

```python
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)
```

## Getting Movie Recommendations

The function get_movie_recommendation(movie_name) returns 10 similar movies for a given movie.

```python
def get_movie_recommendation(movie_name):
    n_movies_to_recommend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_recommend+1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist()))[1:],
                                   key=lambda x: x[1])
        recommendations = [movies.iloc[idx]['title'] for idx, _ in rec_movie_indices]
        return recommendations
    else:
        return "Movie not found."
```

## Example Usage

```bash
get_movie_recommendation("Toy Story")
```

### Output
```
['Toy Story 2', 'A Bug's Life', 'Monsters, Inc.', 'Finding Nemo', 'Shrek', 'The Lion King', 'Cars', 'Ratatouille', 'Ice Age', 'Aladdin']
```

## Visualization

- Scatter plots visualize user voting distribution and movie voting distribution.
- Uses Matplotlib and Seaborn for insights into dataset sparsity and user engagement.

## Conclusion

This Movie Recommendation System efficiently predicts similar movies based on user ratings using item-based filtering. It is optimized for scalability, making it suitable for real-world applications with large datasets.
