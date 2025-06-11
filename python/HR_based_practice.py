import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

# 1. 모델 학습 함수
def load_and_train_model(csv_path):
    df = pd.read_csv(csv_path)

    # Feature Engineering
    df['Heart_Range'] = df['Max_BPM'] - df['Resting_BPM']
    df['Relative_Avg'] = df['Avg_BPM'] / df['Max_BPM']
    df['BPM_per_kg'] = df['Avg_BPM'] / df['Weight (kg)']


    features = [
        "Age", "Gender", "Weight (kg)", "Height (m)",
        "Workout_Frequency (days/week)", "Experience_Level", "BMI",
        "Session_Duration (hours)",
        "Heart_Range", "Relative_Avg",
        "BPM_per_kg"  # 추가
    ]
    target = "Intensity_Label"

    df = shuffle(df, random_state=42)
    X = df[features]
    y = df[target]

    numeric_features = [f for f in features if f != "Gender"]
    categorical_features = ["Gender"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    X_train_processed = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)

    model = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),
    activation='tanh',  # 또는 'relu'
    solver='adam',      # 또는 'sgd'
    alpha=0.0001,
    learning_rate='adaptive',
    max_iter=1500,
    random_state=42
    )

    model.fit(X_train_bal, y_train_bal)

    # 테스트 평가
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

# 2. 실시간 세트 피드백 생성 함수
def generate_set_feedback(user_data_dict, clf):
    user_df = pd.DataFrame([user_data_dict])
    user_df['Heart_Range'] = user_df['Max_BPM'] - user_df['Resting_BPM']
    user_df['Relative_Avg'] = user_df['Avg_BPM'] / user_df['Max_BPM']
    user_df['BPM_per_kg'] = user_df['Avg_BPM'] / user_df['Weight (kg)']



    input_features = [
    "Age", "Gender", "Weight (kg)", "Height (m)",
    "Workout_Frequency (days/week)", "Experience_Level", "BMI",
    "Session_Duration (hours)",
    "Heart_Range", "Relative_Avg",
    "BPM_per_kg"  # 추가
    ]

    user_df = user_df[input_features]
    predicted_intensity = clf.predict(user_df)[0]

    current_weight = user_data_dict["Weight_Used"]
    current_reps = user_data_dict["Repetitions"]
    avg_bpm = user_data_dict["Avg_BPM"]

    if predicted_intensity == "low":
        next_weight = current_weight + 2.5
        next_reps = current_reps
    elif predicted_intensity == "medium":
        next_weight = current_weight
        next_reps = current_reps
    else:  # high
        next_weight = current_weight - 2.5
        next_reps = max(current_reps - 2, 1)

    feedback = (
        f"[1세트 결과 요약]\n"
        f"- 중량: {current_weight}kg, 반복 횟수: {current_reps}회\n"
        f"- 평균 심박수: {avg_bpm}bpm\n"
        f"- 예측된 운동 강도: {predicted_intensity.upper()}\n\n"
        f"[2세트 추천]\n"
        f"- 중량: {next_weight}kg, 반복 횟수: {next_reps}회\n"
        f"=> 운동 강도에 따라 조정된 세트입니다."
    )
    return feedback

# 3. 실행 예시
if __name__ == "__main__":
    model = load_and_train_model("Labeled_with_Intensity_Levels.csv")

    # 사용자 1세트 기록
    sample_user = {
        "Age": 27,
        "Gender": "Male",
        "Weight (kg)": 92,
        "Height (m)": 1.81,
        "Workout_Frequency (days/week)": 4,
        "Experience_Level": 3,
        "BMI": 24.2,
        "Session_Duration (hours)": 0.25,
        "Max_BPM": 190,
        "Resting_BPM": 65,
        "Avg_BPM": 175,  # 높은 평균 심박수
        "Weight_Used": 100,  # 높은 중량
        "Repetitions": 15
    }

    result = generate_set_feedback(sample_user, model)
    print("\n", result)
