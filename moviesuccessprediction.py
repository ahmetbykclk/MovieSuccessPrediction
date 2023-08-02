import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the dataset with tab-separated values
df = pd.read_csv('dataset.csv', sep=',')

# Total null values present in each column
print(df.isnull().sum())

# Droping the samples that have missing values
df.dropna(inplace=True)

# Dropping 2 columns
df.drop(columns=['movie_imdb_link'], inplace=True)

# Label encoding the categorical columns
le = LabelEncoder()
cat_list = ['color', 'director_name', 'actor_2_name', 'genres', 'actor_1_name', 'actor_3_name', 'plot_keywords', 'language', 'country', 'content_rating', 'title_year', 'aspect_ratio']
df[cat_list] = df[cat_list].apply(lambda x: le.fit_transform(x))

# Removing few columns due to multicollinearity
df.drop(columns=['cast_total_facebook_likes', 'num_critic_for_reviews'], inplace=True)

# Prepare features (X) and target (y)
X = df.drop(columns=['imdb_score', 'movie_title'])  # Remove movie_title from features
y = df['imdb_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (Linear Regression)
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Classify predicted IMDb scores into "Flop," "Average," or "Hit"
classify_dict = {0: 'Flop', 1: 'Average', 2: 'Hit'}
y_pred_class = np.digitize(y_pred, [1.5, 3.0, 6.0, 9.3]) - 1
y_pred_class = np.vectorize(classify_dict.get)(y_pred_class)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Print the first 10 test predictions, actual values, and movie titles
print("\nFirst 10 Test Predictions, Actual Values, and Movie Titles:")
for i in range(10):
    movie_title = df.loc[y_test.index[i], 'movie_title']
    actual_imdb_score = y_test.iloc[i]
    predicted_imdb_score = y_pred[i]
    predicted_class = y_pred_class[i]
    print(f"Movie Title: {movie_title}\tPredicted IMDb Score: {predicted_imdb_score:.2f}\tActual IMDb Score: {actual_imdb_score:.2f}\tClassification: {predicted_class}")

color_map = {'Flop': 'red', 'Average': 'green', 'Hit': 'blue'}

# Map the classification labels to colors, handling 'None' with a default color
colors = [color_map[label] if label in color_map else 'gray' for label in y_pred_class]

# Visualize predicted IMDb scores and their classifications
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c=colors, alpha=0.5)
plt.plot([1.5, 9.3], [1.5, 9.3], 'k--')  # Identity line (y = x)
plt.xlabel('Actual IMDb Score')
plt.ylabel('Predicted IMDb Score')
plt.title('Actual IMDb Score vs. Predicted IMDb Score (Color: Classification)')
plt.show()