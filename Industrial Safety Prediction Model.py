import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# 데이터 로드
file_path = 'data/PredictSafe.csv'
data = pd.read_csv(file_path)

# 데이터 전처리 (IQR 방식으로 이상치 처리)
for column in data.columns:
    if data[column].dtype != 'object':  # 숫자형 데이터에 대해서만 처리
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        high_limit = Q3 + 1.5 * IQR
        low_limit = Q1 - 1.5 * IQR
        data = data[(data[column] <= high_limit) & (data[column] >= low_limit)]

# 데이터 정보 및 통계 확인
print("데이터 정보:")
print(data.info())
print("\n데이터 통계 요약:")
print(data.describe())

# 독립 변수와 종속 변수 설정
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Machine failure'

X = data[features]
y = data[target]

# 데이터 시각화 (상관 관계 분석)
correlation_matrix = data[features + [target]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')

# 'results' 파일에 시각화된 그래프 저장
plt.savefig('results.png')
plt.close()  # 그래프를 저장한 후 창을 닫아 메모리 절약

# 훈련 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 머신러닝 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 성능 평가 그래프 (accuracy_graph.png 저장)
plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='green')
plt.ylim(0, 1)
plt.title('Model Accuracy')
plt.ylabel('Accuracy')

# 'results' 파일에 성능 평가 그래프 저장
plt.savefig('results.png', dpi=300, format='png')
plt.close()  # 그래프를 저장한 후 창을 닫아 메모리 절약

# 혼동 행렬 출력
conf_matrix = confusion_matrix(y_test, y_pred)
print('\nConfusion Matrix:')
print(conf_matrix)

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# 'results' 파일에 혼동 행렬 시각화 저장
plt.savefig('results.png', dpi=300, format='png')
plt.close()  # 그래프를 저장한 후 창을 닫아 메모리 절약
