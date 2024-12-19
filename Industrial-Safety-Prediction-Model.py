import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import os

# 데이터 로드
file_path = 'data/PredictSafe.csv'
data = pd.read_csv(file_path)

# 데이터 전처리
# 1. 결측치 처리 (결측치가 있는 행 제거)
data = data.dropna()

# 2. 불균형 처리: RandomOverSampler를 사용하여 클래스 불균형 해결
X = data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']]
y = data['Machine failure']

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

# 데이터 시각화 (상관 관계 분석)
correlation_matrix = data[['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]', 'Machine failure']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
os.makedirs('results', exist_ok=True)
plt.savefig('results/correlation_matrix.png')
plt.show()

# KFold 교차 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(X_resampled):
    X_train, X_test = X_resampled.iloc[train_index], X_resampled.iloc[test_index]
    y_train, y_test = y_resampled.iloc[train_index], y_resampled.iloc[test_index]

    # 모델 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)

    # 성능 평가
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred))
    recall_scores.append(recall_score(y_test, y_pred))
    f1_scores.append(f1_score(y_test, y_pred))

# KFold 평균 성능 계산
avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)
avg_f1 = sum(f1_scores) / len(f1_scores)

# KFold 결과 출력
print("KFold Cross Validation Results:")
print(f"Average Accuracy: {avg_accuracy:.2f}")
print(f"Average Precision: {avg_precision:.2f}")
print(f"Average Recall: {avg_recall:.2f}")
print(f"Average F1 Score: {avg_f1:.2f}")

# Feature Importances 그래프
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importances')
os.makedirs('results', exist_ok=True)
plt.savefig('results/feature_importances.png')
plt.show()

# 성능 평가 그래프
plt.figure(figsize=(8, 6))
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [avg_accuracy, avg_precision, avg_recall, avg_f1], color=['green', 'blue', 'orange', 'red'])
plt.ylim(0, 1)
plt.title('Average Model Performance Metrics')
plt.ylabel('Score')
os.makedirs('results', exist_ok=True)
plt.savefig('results/average_performance_metrics_graph.png')
plt.show()

# 혼동 행렬 그래프 (마지막 KFold 결과 사용)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Last Fold)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('results/confusion_matrix.png')
plt.show()

# 최종 코드 저장
os.makedirs('final exam', exist_ok=True)
final_code_path = 'final exam/Industrial-Safety-Prediction-Model.py'