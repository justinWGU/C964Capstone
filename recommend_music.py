import joblib
import sys

# Load the trained model
model = joblib.load('music_recommendation_model.pkl')

def recommend(age, gender):
    try:
        user_profile = [[age, gender]]
        recommendation = model.predict(user_profile)
        return f"Recommended Genre: {recommendation[0]}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python recommend_music.py <age> <gender>")
        sys.exit(1)

    age = int(sys.argv[1])
    gender = int(sys.argv[2])
    result = recommend(age, gender)
    print(result)