import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


parser = argparse.ArgumentParser(
                    prog='logistic-regression',
                    description='',
                    epilog='')

parser.add_argument("-i", "--input")

args = parser.parse_args()
input_path = args.input or "data/sample.csv"

print(f"Working with file {input_path}...")

df = pd.read_csv(input_path)
df["Date of Transfer"] = pd.to_datetime(df['Date of Transfer'])
df["Transfer Age"] = (pd.Timestamp("2024-01-01") - df['Date of Transfer']) * (-1)
df["Transfer Age In Years"] = df['Transfer Age'] / pd.Timedelta('365 days')
df["IsNew"] = df["Old/New"].replace({'Y': 1, 'N': 0})
df["IsDetached"] = df["Property Type"].replace({'T': 0, 'S': 0, "D": 1, "F": 0, "O": 0})
df["Price Scaled"] = df["Price"] / 10_000

X = df[["Transfer Age In Years", "Price Scaled"]].values
y = df["IsDetached"].astype('int')

norm = tf.keras.layers.Normalization(axis=-1)
norm.adapt(X)
Xn = norm(X)

# single layer neural network - same as logistic regression!
model = Sequential([
    tf.keras.Input(shape=(2,)),
    #Dense(2, activation="sigmoid"),
    Dense(1, activation="sigmoid")
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
)

model.fit(Xn, y, epochs=5)

# TODO add > 0.5 test to make the classification obvious
print("Predict -15, 10: ", model.predict(norm(np.array([-15, 10]).reshape(1, -1))))
print("Predict -15, 20: ", model.predict(norm(np.array([-15, 20]).reshape(1, -1))))
print("Predict -15, 30: ", model.predict(norm(np.array([-15, 30]).reshape(1, -1))))
print("Predict -15, 40: ", model.predict(norm(np.array([-15, 40]).reshape(1, -1))))
print("Predict -15, 50: ", model.predict(norm(np.array([-15, 50]).reshape(1, -1))))
print("Predict -15, 60: ", model.predict(norm(np.array([-15, 60]).reshape(1, -1))))
print("Predict -15, 70: ", model.predict(norm(np.array([-15, 70]).reshape(1, -1))))
print("Predict -15, 80: ", model.predict(norm(np.array([-15, 80]).reshape(1, -1))))

print("Done.")
