import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

# Load and preprocess dataset
def load_and_train_model(csv_path):
    print("Loading and preprocessing dataset...")
    df = pd.read_csv(csv_path)

    # Labeling (별도 수작업 라벨링된 컬럼이 있다고 가정)
    target = "Intensity_Label"  # CSV 파일에 미리 정의된 라벨 컬럼 사용

    features = [
        "Age", "Gender", "Weight (kg)", "Height (m)",
        "Resting_BPM", "Workout_Frequency (days/week)",
        "Experience_Level", "BMI"
    ]

    X = df[features]
    y = df[target]

    # Preprocessing
    numeric_features = [f for f in features if f != "Gender"]
    categorical_features = ["Gender"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Building MLP model pipeline...")
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))
    ])

    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print("Training MLP model WITHOUT Intensity_Index, Relative_Avg, Calories_per_Hour, and SMOTE...")
    clf.fit(X_train, y_train)

    print("\nEvaluating model on test data...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nClassification Report on Test Set:")
    print(classification_report(y_test, y_pred))
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

    return clf

# Predict on new user input
def predict_intensity(clf, user_data_dict):
    user_df = pd.DataFrame([user_data_dict])

    input_features = [
        "Age", "Gender", "Weight (kg)", "Height (m)",
        "Resting_BPM", "Workout_Frequency (days/week)",
        "Experience_Level", "BMI"
    ]
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
        "Resting_BPM": 65,
        "Workout_Frequency (days/week)": 3,
        "Experience_Level": 2,
        "BMI": 23.5
    }
    result = predict_intensity(model, sample_user)
    print(f"Predicted workout intensity level: {result}")