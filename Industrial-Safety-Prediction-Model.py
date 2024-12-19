import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

# 훈련 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 머신러닝 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Feature Importances 그래프
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 6))
feature_importances.sort_values(ascending=False).plot(kind='bar', title='Feature Importances')
os.makedirs('results', exist_ok=True)
plt.savefig('results/feature_importances.png')
plt.show()

# 성능 평가 및 혼동 행렬 그래프
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(['Accuracy', 'Precision', 'Recall', 'F1 Score'], [accuracy, precision, recall, f1], color=['green', 'blue', 'orange', 'red'])
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.ylabel('Score')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/performance_and_confusion_matrix.png')
plt.show()

# 최종 코드 저장
os.makedirs('final exam', exist_ok=True)
final_code_path = 'final exam/Industrial-Safety-Prediction-Model.py'