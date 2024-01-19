import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression 

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

reg = LogisticRegression(max_iter=1000).fit(X, y)

print("Predict -15, 10: ", reg.predict(np.array([-15, 10]).reshape(1, -1)))
print("Predict -15, 20: ", reg.predict(np.array([-15, 20]).reshape(1, -1)))
print("Predict -15, 30: ", reg.predict(np.array([-15, 30]).reshape(1, -1)))
print("Predict -15, 40: ", reg.predict(np.array([-15, 40]).reshape(1, -1)))
print("Predict -15, 50: ", reg.predict(np.array([-15, 50]).reshape(1, -1)))
print("Predict -15, 60: ", reg.predict(np.array([-15, 60]).reshape(1, -1)))
print("Predict -15, 70: ", reg.predict(np.array([-15, 70]).reshape(1, -1)))
print("Predict -15, 80: ", reg.predict(np.array([-15, 80]).reshape(1, -1)))

print("Score: ", reg.score(X, y))

groups = df.groupby('IsDetached')

for name, group in groups:
    plt.plot(group["Transfer Age In Years"], group["Price Scaled"], linestyle='', marker='o', ms=5, label=name)

# plot the decision boundary
a0 = reg.intercept_[0]
a1 = reg.coef_[0][0]
a2 = reg.coef_[0][1]
x1 = np.array([-30, 0])
x2 = (-a0 - (a1 * x1)) / a2
plt.plot(x1, x2, color = "red")

# some houses have incredibly high prices
# limit the plot to 2*standard deviation
plt.ylim(0, 5 * df["Price Scaled"].std())
plt.xlim(-30, 0)

plt.show()

print("Done.")