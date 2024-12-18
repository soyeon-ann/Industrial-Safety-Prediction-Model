import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import os
import json

# 데이터 로드
data = pd.read_csv('data/PredictSafe_data.csv')

# 데이터 첫 5개 샘플 확인
print(data.head())

# 결측값 처리
data = data.dropna()  # 결측값 있는 행 제거

# 범주형 변수 처리 (예: 레이블 인코딩)
data['category'] = data['category'].astype('category').cat.codes

# 독립변수(X)와 종속변수(y) 설정
X = data.drop('target', axis=1)  # 'target' 열을 제외한 나머지는 독립 변수
y = data['target']  # 'target' 열은 종속 변수

# 훈련 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화 (정규화)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 랜덤 포레스트 분류기 모델 생성
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 예측 수행
y_pred = model.predict(X_test)

# 모델 성능 평가
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix 시각화
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()

# 시각화된 결과를 results 폴더에 저장
if not os.path.exists('results'):
    os.makedirs('results')

plt.savefig('results/Confusion_Matrix.png')

# 모델 성능 비교 (KFold 교차 검증)
cv_scores = cross_val_score(model, X, y, cv=5)
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean Cross-Validation Score: {np.mean(cv_scores)}")

# 특성 중요도 시각화
feature_importance = model.feature_importances_
indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importance')
plt.bar(range(X.shape[1]), feature_importance[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()

# 특성 중요도 시각화 결과 저장
plt.savefig('results/Feature_Importance.png')

# 결과 파일에 저장
results = {
    "classification_report": classification_report(y_test, y_pred),
    "confusion_matrix": cm.tolist(),
    "cross_val_scores": cv_scores.tolist(),
    "mean_cross_val_score": np.mean(cv_scores)
}

# results 폴더에 결과 저장
if not os.path.exists('results'):
    os.makedirs('results')

with open('results/safety_model_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("모든 결과가 'results' 폴더에 저장되었습니다.")
