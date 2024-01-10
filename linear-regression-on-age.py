import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 

parser = argparse.ArgumentParser(
                    prog='linear-regression-on-age',
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

# need to reshape since we only have one feature
X = np.reshape(df["Transfer Age In Years"], (-1,1))
y = df["Price"]

reg = LinearRegression().fit(X, y)

print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

print(f"Prediction for a house sold 30 years ago is\t£{reg.predict([[-30]])[0]:,.0f}")
print(f"Prediction for a house sold 20 years ago is\t£{reg.predict([[-20]])[0]:,.0f}")
print(f"Prediction for a house sold 10 years ago is\t£{reg.predict([[-10]])[0]:,.0f}")
print(f"Prediction for a house sold in 2024 is     \t£{reg.predict([[0]])[0]:,.0f}")

y_prediction = X * reg.coef_ + reg.intercept_

df.plot.scatter("Transfer Age In Years", "Price")
plt.plot(X, y_prediction, color = "red")

# some houses have incredibly high prices
# limit the plot to 2*standard deviation
plt.ylim(0, 2 * df["Price"].std())
plt.show()

print("Done.")