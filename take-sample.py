import random
import argparse

parser = argparse.ArgumentParser(
                    prog='take-sample',
                    description='Takes a random sample from a CSV file',
                    epilog='')

parser.add_argument("-i", "--input")
parser.add_argument("-o", "--output")
parser.add_argument("-c", "--count")

args = parser.parse_args()
sample_size = int(args.count or 1000)
input_path = args.input or "data/price_paid_records.csv"
output_path = args.output or "data/sample.csv"

print(f"Copying a random sample of {sample_size} data points from {input_path} to {output_path}...")

with open(input_path, "r") as input_file:
    lines = input_file.readlines()
    total_number_of_lines = len(lines)
    header = lines[0]
    
    with open(output_path, "w") as sample_file:
        sample_file.write(header)
        
        for i in range(1, sample_size):
            random_index = random.randint(0, total_number_of_lines)
            random_line = lines[random_index]
            sample_file.write(random_line)
            
print(f"Sample successful. {output_path}")
