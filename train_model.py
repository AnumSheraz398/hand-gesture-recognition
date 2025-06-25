import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import joblib

# Load the data
df = pd.read_csv("gesture_data.csv")

# Step 2: Split data and labels
X = df.drop("label", axis=1).values
y = df["label"].values

# Step 3: Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# Step 5: Define model
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(X.shape[1],)))
model.add(Dense(64, activation="relu"))
model.add(Dense(len(np.unique(y)), activation="softmax"))

# Step 6: Compile and train
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Step 7: Save model and label encoder
model.save("gesture_model.h5")
joblib.dump(label_encoder, "label_encoder.pkl")

print("✅ Model trained and saved successfully.")