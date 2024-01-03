# text processing
# read text from files, must be in a format that connects some sort of situation to a solution, or series of techniques to solve the sitaution:
# files will be stored in large repository, but for now we can create a fake, but applicable one:

situation = ""
solution = "" # placeholder, the term "solution" is probably not the best term

import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
# from tensorflow.keras import layers, models # issue with current version of tensorflow
# from tensorflow.python.keras import layers, models
from keras.api._v2.keras import layers, models

# 1.
# =========================================================== #
data = []
def build_problems(directory_str, data_list):
    for item in directory_str:
        parse info (situation)
        parse "solution"

        send to data_list as tuple (ordered pair but can expand if needed)
    return data_list

# data now is a large list of points:
# (situation, solution)
# =========================================================== #


# 2.
# =========================================================== #
# NLP word embedding (using spaCy):
# following function will turn text into a vector of numbers (which is the only thing a NN can read)
def extract_features(text):
    doc = nlp(text)
    return doc.vector
# =========================================================== #


# 3.
# =========================================================== #
# more data processing:
# the following seperates all datapoints into 
# 1. a group that will be used to train the model (so it knows whats right)
# 2. a group used for testing (we can also create our own, common practice is to split datasets into 80% training, 20% testing)

# from my final project, import statement predictor
X = np.array([extract_features(problem["text"]) for problem in problems])
y = np.array([1 if "import" in problem["code"] else 0 for problem in problems])
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# pseudocode:
group1 = numpy array of (NLP processed situation)
group2 = numpy array of (NLP processed solution)
split
# =========================================================== #


# 4.
# =========================================================== #
# send to model

# Define a simple neural network model
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
model.add(layers.Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
# =========================================================== #


# 5.
# =========================================================== #
# evaluate using the test set (seperated and saved in step 3)
_, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
# =========================================================== #