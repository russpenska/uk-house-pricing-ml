import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression 

parser = argparse.ArgumentParser(
                    prog='filtering',
                    description='',
                    epilog='')

parser.add_argument("-i", "--input")
parser.add_argument("-o", "--output")

args = parser.parse_args()
input_path = args.input or "data/sample.csv"
output_path = args.output or "data/bristol.csv"

print(f"Working with file {input_path}...")

df = pd.read_csv(input_path)

df = df[df["Town/City"] == "BRISTOL"]
df.to_csv(output_path, index=False)