import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Load and preprocess dataset
def load_and_train_model(csv_path):
    print("Loading and preprocessing dataset...")
    df = pd.read_csv(csv_path)

    # Feature Engineering
    df['Heart_Range'] = df['Max_BPM'] - df['Resting_BPM']
    df['Relative_Avg'] = df['Avg_BPM'] / df['Max_BPM']
    df['Calories_per_Hour'] = df['Calories_Burned'] / df['Session_Duration (hours)']
    df['Intensity_Index'] = (df['Avg_BPM'] - df['Resting_BPM']) / (df['Max_BPM'] - df['Resting_BPM'])

    # New Labeling using Intensity Index
    def label_from_intensity_index(i):
        if i < 0.6:
            return 'low'
        elif i < 0.8:
            return 'medium'
        else:
            return 'high'

    df['Intensity_Label'] = df['Intensity_Index'].apply(label_from_intensity_index)

    # IMPORTANT: Remove 'Intensity_Index' from features to avoid leakage
    features = [
        "Age", "Gender", "Weight (kg)", "Height (m)",
        "Resting_BPM", "Workout_Frequency (days/week)",
        "Experience_Level", "BMI", "Heart_Range",
        "Relative_Avg", "Calories_per_Hour"
    ]
    target = "Intensity_Label"

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

    print("Building Random Forest model pipeline...")
    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
    ])

    # Train/test split
    print("Splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Apply preprocessing to training data before SMOTE
    print("Applying preprocessing to training data for SMOTE...")
    X_train_processed = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)

    # Recreate classifier-only pipeline for balanced training
    print("Training Random Forest model with SMOTE-balanced data...")
    clf.named_steps['classifier'].fit(X_train_bal, y_train_bal)

    # Evaluate on test data (apply full pipeline to X_test)
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

    # 자동으로 파생 feature 계산
    user_df['Heart_Range'] = user_df['Max_BPM'] - user_df['Resting_BPM']
    user_df['Relative_Avg'] = user_df['Avg_BPM'] / user_df['Max_BPM']
    user_df['Calories_per_Hour'] = user_df['Calories_Burned'] / user_df['Session_Duration (hours)']

    # 모델 입력 feature만 선택
    input_features = [
        "Age", "Gender", "Weight (kg)", "Height (m)",
        "Resting_BPM", "Workout_Frequency (days/week)",
        "Experience_Level", "BMI", "Heart_Range",
        "Relative_Avg", "Calories_per_Hour"
    ]
    user_df = user_df[input_features]

    prediction = clf.predict(user_df)[0]
    return prediction

if __name__ == "__main__":
    # Step 1: Train model with your dataset
    model = load_and_train_model("Labeled_with_Intensity_Levels.csv")

    # Step 2: Predict on new data
    sample_user = {
        "Age": 35,
        "Gender": "Male",
        "Weight (kg)": 72,
        "Height (m)": 1.75,
        "Resting_BPM": 65,
        "Workout_Frequency (days/week)": 3,
        "Experience_Level": 2,
        "BMI": 23.5,
        "Avg_BPM": 157,
        "Max_BPM": 190,
        "Calories_Burned": 400,
        "Session_Duration (hours)": 0.5
    }
    result = predict_intensity(model, sample_user)
    print(f"Predicted workout intensity level: {result}")
