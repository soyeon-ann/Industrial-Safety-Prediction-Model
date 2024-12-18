"""Final Industrial Safety Prediction Model"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ������ �ε�
file_path = 'data/PredictSafe.csv'
data = pd.read_csv(file_path)

# ���� ������ ���� ���� ����
features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
target = 'Machine failure'

X = data[features]
y = data[target]

# �Ʒ� �����Ϳ� �׽�Ʈ �����ͷ� ������
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# �ӽŷ��� �� �н�
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ����
y_pred = model.predict(X_test)

# ���� ��
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# ȥ�� ��� ���
conf_matrix = confusion_matrix(y_test, y_pred)
print('
Confusion Matrix:')
print(conf_matrix)

# ȥ�� ��� �ð�ȭ
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('results/confusion_matrix.png')
plt.show()
