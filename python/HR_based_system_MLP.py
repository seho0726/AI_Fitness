import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

# Load and preprocess dataset
def load_and_train_model(csv_path):
    print("Loading and preprocessing dataset...")
    df = pd.read_csv(csv_path)

    if 'Intensity_Label' not in df.columns:
        raise ValueError("CSV must contain 'Intensity_Label' column without computing it from heart rate data.")

    # Optional feature engineering if data available
    if all(col in df.columns for col in ['Max_BPM', 'Resting_BPM']):
        df['Heart_Range'] = df['Max_BPM'] - df['Resting_BPM']
    if all(col in df.columns for col in ['Avg_BPM', 'Max_BPM']):
        df['Relative_Avg'] = df['Avg_BPM'] / df['Max_BPM']
    if all(col in df.columns for col in ['Calories_Burned', 'Session_Duration (hours)']):
        df['Calories_per_Hour'] = df['Calories_Burned'] / df['Session_Duration (hours)']

    features = [
        "Age", "Gender", "Weight (kg)", "Height (m)",
        "Workout_Frequency (days/week)", "Experience_Level", "BMI",
        "Calories_Burned", "Session_Duration (hours)",
        "Heart_Range", "Relative_Avg", "Calories_per_Hour"
    ]
    features = [f for f in features if f in df.columns]
    target = "Intensity_Label"

    df = shuffle(df, random_state=42)
    X = df[features]
    y = df[target]

    numeric_features = [f for f in features if f != "Gender"]
    categorical_features = ["Gender"] if "Gender" in features else []

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Building MLP model pipeline...")
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=1000, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Applying preprocessing to training data for SMOTE...")
    X_train_processed = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)

    print("Training the MLP model...")
    pipe.named_steps['classifier'].fit(X_train_bal, y_train_bal)

    X_test_processed = preprocessor.transform(X_test)
    y_pred = pipe.named_steps['classifier'].predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred))
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Full pipeline (preprocessor + trained classifier)
    pipe.fit(X_train, y_train)
    return pipe

# Predict on new user input
def predict_intensity(clf, user_data_dict):
    user_df = pd.DataFrame([user_data_dict])

    if 'Max_BPM' in user_data_dict and 'Resting_BPM' in user_data_dict:
        user_df['Heart_Range'] = user_data_dict['Max_BPM'] - user_data_dict['Resting_BPM']
    if 'Avg_BPM' in user_data_dict and 'Max_BPM' in user_data_dict:
        user_df['Relative_Avg'] = user_data_dict['Avg_BPM'] / user_data_dict['Max_BPM']
    if 'Calories_Burned' in user_data_dict and 'Session_Duration (hours)' in user_data_dict:
        user_df['Calories_per_Hour'] = user_data_dict['Calories_Burned'] / user_data_dict['Session_Duration (hours)']

    input_features = [
        "Age", "Gender", "Weight (kg)", "Height (m)",
        "Workout_Frequency (days/week)", "Experience_Level", "BMI",
        "Calories_Burned", "Session_Duration (hours)",
        "Heart_Range", "Relative_Avg", "Calories_per_Hour"
    ]
    input_features = [f for f in input_features if f in user_df.columns]
    user_df = user_df[input_features]
    prediction = clf.predict(user_df)[0]
    return prediction

if __name__ == "__main__":
    model = load_and_train_model("Labeled_with_Intensity_Levels.csv")

    sample_user = {
        "Age": 35,
        "Gender": "Male",
        "Weight (kg)": 72,
        "Height (m)": 1.75,
        "Workout_Frequency (days/week)": 3,
        "Experience_Level": 2,
        "BMI": 23.5,
        "Calories_Burned": 400,
        "Session_Duration (hours)": 0.5,
        "Max_BPM": 190,
        "Resting_BPM": 70,
        "Avg_BPM": 130
    }
    result = predict_intensity(model, sample_user)
    print(f"Predicted workout intensity level: {result}")
