import pandas as pd
import numpy as np
import tensorflow as tf

import mlflow.tensorflow
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error

tf.keras.utils.set_random_seed(42)

batch_size = 128
epochs = 100

new_project_root = str(Path(__file__).parent.parent / "data" / "salary_prediction_data_redacted")
print(new_project_root)

df = pd.read_csv(new_project_root)
print("CSV прочитан:", df.shape, flush=True)

X = df.drop(["Salary"], axis=1)
y = df["Salary"].astype(np.float32)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("name")

# Подготовка числовых и категориальных признаков
numeric_features = ["Experience", "Age"]
categorical_features = ["Education", "Location", "Job_Title", "Gender"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(
    np.asarray(y_train).reshape(-1, 1)
).ravel()

y_val_scaled = target_scaler.transform(
    np.asarray(y_val).reshape(-1, 1)
).ravel()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore",
        sparse_output=False), categorical_features),
])

X_train = preprocessor.fit_transform(X_train).astype(np.float32)
X_val = preprocessor.transform(X_val).astype(np.float32)
X_test = preprocessor.transform(X_test).astype(np.float32)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1),
])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="mse",
    metrics=[
        tf.keras.metrics.MeanAbsoluteError(name="mae"),
        tf.keras.metrics.RootMeanSquaredError(name="rmse"),
    ],
)

model.summary()

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="mlmodels/model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="min"
)

with mlflow.start_run(run_name="run_name"):

    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "metrics": ["MAE","RMSE"],
        "loss": "MSE",

    })

    history = model.fit(
        X_train,
        y_train_scaled,
        validation_data=(X_val, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, early_stopping],
        verbose=1,
    )

    print(history.history.keys())

    for epoch in range(len(history.history["loss"])):

        mlflow.log_metrics({
            "train_loss": history.history["loss"][epoch],
            "train_rmse": history.history["rmse"][epoch],
            "train_mae": history.history["mae"][epoch],
            "val_loss": history.history["val_loss"][epoch],
            "val_mae": history.history["val_mae"][epoch],
            "val_rmse": history.history["val_rmse"][epoch]
        }, step=epoch + 1)

    best_model = tf.keras.models.load_model("mlmodels/model.keras")

    # Оценка в стандартизированном масштабе
    y_test_scaled = target_scaler.transform(
        np.asarray(y_test).reshape(-1, 1)
    ).ravel()

    scaled_loss, scaled_mae, scaled_rmse = best_model.evaluate(
        X_test,
        y_test_scaled,
        verbose=0
    )

    mlflow.log_metrics({"test_mse": scaled_loss,
                        "test_mae": scaled_mae,
                        "test_rmse": scaled_rmse
                        })

    mlflow.log_artifact("mlmodels/model.keras")
    print(scaled_loss, scaled_mae, scaled_rmse)