from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV # הפונקציה שעושה אופטימיזציה ואימות צולב
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_and_evaluate_model(x_train, y_train, x_test, y_test):
        
    base_model = MultinomialNB()

    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    }

    grid_search = GridSearchCV(
        estimator=base_model, 
        param_grid=param_grid, 
        cv=5, 
        scoring='f1'
    )
    
    grid_search.fit(x_train, y_train)
    
    best_model = grid_search.best_estimator_
    
    print(f"Optimized params {grid_search.best_params_}\n")
    
    y_prediction_train = best_model.predict(x_train)
    y_prediction_test = best_model.predict(x_test)
    
    metrics = {
        "Train" : {
            "Accuracy": accuracy_score(y_train, y_prediction_train),
            "Precision": precision_score(y_train, y_prediction_train, pos_label=1),
            "Recall": recall_score(y_train, y_prediction_train, pos_label=1),
            "F1-Score": f1_score(y_train, y_prediction_train, pos_label=1)
        },
        "Test": {
            "Accuracy": accuracy_score(y_test, y_prediction_test),
            "Precision": precision_score(y_test, y_prediction_test, pos_label=1),
            "Recall": recall_score(y_test, y_prediction_test, pos_label=1),
            "F1-Score": f1_score(y_test, y_prediction_test, pos_label=1)
        }
    }
    
    print("-" * 40)
    print("Train Result (Using Best Naive Bayes Model):")
    print(f"Accuracy:  {metrics['Train']['Accuracy']:.4f}")
    print(f"Precision: {metrics['Train']['Precision']:.4f}")
    print(f"Recall:    {metrics['Train']['Recall']:.4f}")
    print(f"F1-Score:  {metrics['Train']['F1-Score']:.4f}")
    print("-" * 40)
    
    print("Test Result (Using Best Naive Bayes Model):")
    print(f"Accuracy:  {metrics['Test']['Accuracy']:.4f}")
    print(f"Precision: {metrics['Test']['Precision']:.4f}")
    print(f"Recall:    {metrics['Test']['Recall']:.4f}")
    print(f"F1-Score:  {metrics['Test']['F1-Score']:.4f}")
    print("-" * 40)
    
    return best_model, metrics