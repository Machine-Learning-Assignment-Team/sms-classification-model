# קובץ main.py זמני לבדיקה
import numpy as np
from model_logic import train_and_evaluate_model

print("מייצר נתונים שיקריים לחלוטין לבדיקת הלוגיקה")
X_train_dummy = np.random.rand(100, 50) 
y_train_dummy = np.random.randint(2, size=100) # 0 או 1

X_test_dummy = np.random.rand(20, 50)
y_test_dummy = np.random.randint(2, size=20)

trained_model, results = train_and_evaluate_model(
    X_train_dummy, y_train_dummy, X_test_dummy, y_test_dummy
)