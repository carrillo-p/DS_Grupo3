-- Script para crear la tabla patient_predictions

CREATE TABLE IF NOT EXISTS patient_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    hypertension INT,
    heart_disease INT,
    ever_married VARCHAR(5),
    work_type VARCHAR(50),
    Residence_type VARCHAR(10),
    smoking_status VARCHAR(20),
    bmi_category VARCHAR(20),
    age_category VARCHAR(20),
    glucose_level_category VARCHAR(20),
    prediction INT,
    prediction_probability FLOAT
);