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

def print_help():
    help_text = """
    Music Recommendation System

    Usage:
        python recommend_music.py <age> <gender>

    Example:
        python recommend_music.py 21 1

    Arguments:
        <age>    : Age of the user (integer)
        <gender> : Gender of the user (1 for male, 0 for female)

    Options:
        -h, --help : Show this help message and exit
    """
    print(help_text)

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] in ('-h', '--help'):
        print_help()
        sys.exit(0)

    if len(sys.argv) != 3:
        print("Error: Invalid number of arguments.")
        print_help()
        sys.exit(1)

    try:
        age = int(sys.argv[1])
        gender = int(sys.argv[2])
        if gender not in [0, 1]:
            raise ValueError("Gender must be 0 (female) or 1 (male).")
    except ValueError as e:
        print(f"Error: {e}")
        print_help()
        sys.exit(1)

    result = recommend(age, gender)
    print(result)