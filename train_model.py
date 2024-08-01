import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import matplotlib.pyplot as plt

# Load data
music_data = pd.read_csv('music.csv')

# Split data into input (X) and output (y) sets
X = music_data.drop(columns=['genre'])
y = music_data['genre']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'music_recommendation_model.pkl')

# Descriptive statistics and visualizations
print(music_data.describe())

# Bar Plot
music_data['genre'].value_counts().plot(kind='bar', title='Genre Distribution')
plt.xlabel('Genre')
plt.ylabel('Count')
plt.show()

# Scatterplot
plt.scatter(music_data['age'], music_data['genre'].apply(lambda x: {'hiphop': 1, 'jazz': 2, 'classical': 3, 'dance': 4, 'acoustic': 5}[x]))
plt.title('Age vs Genre')
plt.xlabel('Age')
plt.ylabel('Genre')
plt.show()

# Pie Chart
music_data['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Gender Distribution')
plt.show()