import numpy as np

def calculate_metrics(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1


class CustomMultinomialNB:
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X, y):
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape
        
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            feature_counts = np.asarray(X_c.sum(axis=0)).flatten()
            
            # Laplace smoothing to prevent zero probabilities
            smoothed_fc = feature_counts + self.alpha
            smoothed_cc = smoothed_fc.sum()
            
            # Using log probabilities to prevent numerical underflow
            self.feature_log_prob_[idx, :] = np.log(smoothed_fc / smoothed_cc)
            
            if self.fit_prior:
                self.class_log_prior_[idx] = np.log(X_c.shape[0] / n_samples)
            else:
                self.class_log_prior_[idx] = np.log(1.0 / n_classes)
                
        return self

    def predict(self, X):
        jll = X.dot(self.feature_log_prob_.T) + self.class_log_prior_
        return self.classes_[np.argmax(jll, axis=1)]


class CustomGridSearch:
    def __init__(self, param_grid, cv=5):
        self.param_grid = param_grid
        self.cv = cv
        self.best_score_ = -1

    def _generate_combinations(self, param_grid):
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = []
        
        def build_combos(current_combo, depth):
            if depth == len(keys):
                combinations.append(dict(current_combo))
                return
            
            current_key = keys[depth]
            for val in values[depth]:
                current_combo[current_key] = val
                build_combos(current_combo, depth + 1)
                
        build_combos({}, 0)
        return combinations

    def fit(self, X, y):
        combinations = self._generate_combinations(self.param_grid)
        
        y = np.array(y)
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        # Simple K-Fold manual split
        fold_sizes = np.full(self.cv, n_samples // self.cv, dtype=int)
        fold_sizes[:n_samples % self.cv] += 1
        
        folds = []
        current = 0
        for size in fold_sizes:
            folds.append((current, current + size))
            current += size

        for params in combinations:
            f1_scores = []
            
            for start, stop in folds:
                test_idx = indices[start:stop]
                train_idx = np.concatenate([indices[:start], indices[stop:]])
                
                model = CustomMultinomialNB(**params)
                model.fit(X[train_idx], y[train_idx])
                preds = model.predict(X[test_idx])
                
                _, _, _, f1 = calculate_metrics(y[test_idx], preds)
                f1_scores.append(f1)
                
            avg_f1 = np.mean(f1_scores)
            
            if avg_f1 > self.best_score_:
                self.best_score_ = avg_f1
                self.best_params_ = params

        # Train final model on all data with the best parameters
        self.best_estimator_ = CustomMultinomialNB(**self.best_params_)
        self.best_estimator_.fit(X, y)


def train_and_evaluate_model(x_train, y_train, x_test, y_test):
    print("Starting Custom Grid Search (CV=5)...")

    param_grid = {
        'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'fit_prior': [True, False]
    }

    grid_search = CustomGridSearch(param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Optimization complete! Best params: {grid_search.best_params_}\n")

    train_preds = best_model.predict(x_train)
    test_preds = best_model.predict(x_test)

    train_acc, train_prec, train_rec, train_f1 = calculate_metrics(y_train, train_preds)
    test_acc, test_prec, test_rec, test_f1 = calculate_metrics(y_test, test_preds)

    metrics = {
        "Train": {
            "Accuracy": train_acc,
            "Precision": train_prec,
            "Recall": train_rec,
            "F1-Score": train_f1
        },
        "Test": {
            "Accuracy": test_acc,
            "Precision": test_prec,
            "Recall": test_rec,
            "F1-Score": test_f1
        }
    }

    print("-" * 40)
    print("Train Result (Optimized Model):")
    print(f"Accuracy:  {train_acc:.4f}")
    print(f"Precision: {train_prec:.4f}")
    print(f"Recall:    {train_rec:.4f}")
    print(f"F1-Score:  {train_f1:.4f}")
    print("-" * 40)

    print("Test Result (Optimized Model):")
    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print("-" * 40)

    return best_model, metrics, grid_search
