import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os

class XGBoostStrokeModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def load_data(self, filepath):
        df = pd.read_csv(filepath)
        self.feature_names = df.drop('stroke', axis=1).columns
        print("Columnas del dataset:")
        print(df.columns)
        return df

    def preprocess_data(self, df):
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        X_scaled = self.scaler.fit_transform(X)
        print("Características después del preprocesamiento:")
        print(pd.DataFrame(X_scaled, columns=X.columns).head())
        return X_scaled, y

    def train_model(self, X, y, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

            self.model = xgb.XGBClassifier(
                n_estimators=179,
                learning_rate=0.07,
                max_depth=9,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.92,
                random_state=42
            )
            self.model.fit(X_train_resampled, y_train_resampled)

            y_test_pred = self.model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            accuracies.append(test_accuracy)

            self.evaluate_model(X_test, y_test)

        average_accuracy = sum(accuracies) / len(accuracies)
        print(f"\nAverage Accuracy across folds: {average_accuracy:.4f}")

    def evaluate_model(self, X_test, y_test):
        y_test_pred = self.model.predict(X_test)
        y_test_pred_proba = self.model.predict_proba(X_test)[:, 1]

        print("\nClassification report for the test set:")
        print(classification_report(y_test, y_test_pred))

        print("\nConfusion matrix for the test set:")
        print(confusion_matrix(y_test, y_test_pred))

        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        print(f"\nROC AUC Score: {roc_auc:.4f}")

        self.plot_roc_curve(y_test, y_test_pred_proba)
        self.plot_feature_importance()

    def plot_roc_curve(self, y_test, y_test_pred_proba):
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})', color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def plot_feature_importance(self):
        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        print("\nFeature importance:")
        print(importance_df)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance')
        plt.show()

    def save_model(self, model_path, scaler_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    @classmethod
    def load_model(cls, model_path, scaler_path):
        instance = cls()
        instance.model = joblib.load(model_path)
        instance.scaler = joblib.load(scaler_path)
        return instance

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    

if __name__ == "__main__":
    model = XGBoostStrokeModel()
    # Construir la ruta relativa al archivo de datos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'stroke_woe_smote.csv')
    df = model.load_data(data_path)
    X, y = model.preprocess_data(df)
    model.train_model(X, y)
    # Ajustar las rutas para guardar el modelo y el scaler
    model_path = os.path.join(current_dir, 'xgboost_model.joblib')
    scaler_path = os.path.join(current_dir, 'xgb_scaler.joblib')
    model.save_model(model_path, scaler_path)