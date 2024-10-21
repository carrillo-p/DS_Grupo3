import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import mlflow
import mlflow.sklearn
import mysql.connector
from mysql.connector import Error
import time
from threading import Thread
from dotenv import load_dotenv
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

load_dotenv()

mlflow.set_tracking_uri('http://mlflow:5000')
mlflow.set_experiment('stroke_prediction_xgboost')

class XGBoostStrokeModel:
    def __init__(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_names = None
        self.last_check_file = 'last_check.txt'
        self.check_interval = int(os.getenv('CHECK_INTERVAL', 3600))  # 1 hora por defecto, configurable
        self.db_config = {
            'host': os.getenv('DB_HOST', 'mysql'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME'),
            'table': os.getenv('DB_TABLE', 'new_data')
        }

    def create_db_engine(self):
        db_uri = f"mysql+pymysql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}/{self.db_config['database']}"
        return create_engine(db_uri)

    def load_data_from_mysql(self):
        try:
            metadata = MetaData()
            table = Table(self.db_config['table'], metadata, autoload_with=self.engine)
            Session = sessionmaker(bind=self.engine)
            session = Session()
            query = session.query(table)
            df = pd.read_sql(query.statement, self.engine)
            self.feature_names = df.drop('stroke', axis=1).columns
            print(f"Datos cargados. Número de filas: {len(df)}")
            return df
        except Exception as e:
            print(f"Error al cargar datos desde MySQL: {e}")
            return pd.DataFrame()

    def preprocess_data(self, df):
        X = df.drop('stroke', axis=1)
        y = df['stroke']
        X_scaled = self.scaler.transform(X)
        return X_scaled, y

    def retrain_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        with mlflow.start_run():
            self.model.fit(X_train, y_train)
            
            # Registrar parámetros
            mlflow.log_params(self.model.get_params())
            
            # Evaluar y registrar métricas
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
            
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Registrar el modelo
            mlflow.sklearn.log_model(self.model, "model")
            
            # Generar y registrar gráficos
            self.log_plots(X_test, y_test)
            
        return X_test, y_test

    def evaluate_model(self, X_test, y_test, output_dir='output'):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Classification Report
        report = classification_report(y_test, y_pred)
        with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        mlflow.log_artifact(os.path.join(output_dir, 'classification_report.txt'))

        # ROC AUC
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        mlflow.log_metric("roc_auc", roc_auc)

        # Generar y registrar gráficos
        self.log_plots(X_test, y_test, output_dir)

    def log_plots(self, X_test, y_test, output_dir='output'):
        # Confusion Matrix
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        mlflow.log_artifact(os.path.join(output_dir, 'confusion_matrix.png'))

        # ROC Curve
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        mlflow.log_artifact(os.path.join(output_dir, 'roc_curve.png'))

        # Feature Importance
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        mlflow.log_artifact(os.path.join(output_dir, 'feature_importance.png'))

    def save_model(self, model_path, scaler_path):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Modelo guardado en {model_path}")
        print(f"Scaler guardado en {scaler_path}")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(scaler_path)

    def check_and_retrain(self):
        if self.should_check():
            with mlflow.start_run():
                df = self.load_data_from_mysql()
                
                if len(df) > int(os.getenv('RETRAIN_THRESHOLD', 10000)):
                    X, y = self.preprocess_data(df)
                    X_test, y_test = self.retrain_model(X, y)
                    self.save_model('xgboost_model.joblib', 'xgb_scaler.joblib')
                    self.evaluate_model(X_test, y_test)
                else:
                    mlflow.log_param("retrained", False)
                    mlflow.log_param("data_count", len(df))
                
                self.update_last_check()

    def check_and_retrain(self):
        if self.should_check():
            df = self.load_data_from_mysql()
            
            if len(df) > int(os.getenv('RETRAIN_THRESHOLD', 10000)):
                X, y = self.preprocess_data(df)
                self.retrain_model(X, y)
                self.save_model('xgboost_model.joblib', 'xgb_scaler.joblib')
            
            self.update_last_check()

    def should_check(self):
        if not os.path.exists(self.last_check_file):
            return True
        
        with open(self.last_check_file, 'r') as f:
            last_check = float(f.read().strip())
        
        return time.time() - last_check > self.check_interval

    def update_last_check(self):
        with open(self.last_check_file, 'w') as f:
            f.write(str(time.time()))
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
    

def background_worker(model):
    while True:
        model.check_and_retrain()
        time.sleep(model.check_interval)

# Inicialización del modelo y el worker en segundo plano
model_path = 'src/model/xgboost_model.joblib'
scaler_path = 'src/model/xgb_scaler.joblib'
model = XGBoostStrokeModel(model_path, scaler_path)
bg_thread = Thread(target=background_worker, args=(model,))
bg_thread.start()