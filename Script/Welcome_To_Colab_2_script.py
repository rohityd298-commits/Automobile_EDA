# ---- Cell 1 ----
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load dataset (205 rows, 26 columns, no header row, '?' indicates missing values)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
columns = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]

df = pd.read_csv(url, names=columns, na_values="?")

# Convert numeric specification columns stored as strings/objects
for col in ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm", "price"]:
  df[col] = pd.to_numeric(df[col], errors="coerce")

# 2. Required initial inspection
print("Shape (rows, columns):", df.shape)
print("\n--- First 5 rows (head) ---")
print(df.head())

print("\n--- Last 5 rows (tail) ---")
print(df.tail())

print("\n--- Data types and non-null counts (info) ---")
print(df.info())

print("\n--- Summary statistics (describe) ---")
print(df.describe(include="all"))

# ---- Cell 2 ----
# Question 5: Identify columns with missing values and count them
missing_counts = df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0].sort_values(ascending=False)
print("\n--- Question 5: Columns with missing values ---")
print(missing_cols)

# Check duplicate rows
duplicate_count = df.duplicated().sum()
print("\nDuplicate rows count:", duplicate_count)

# Question 2: Average, minimum, and maximum car prices using a NumPy array
valid_prices_arr = df["price"].dropna().to_numpy()
avg_price = np.mean(valid_prices_arr)
min_price = np.min(valid_prices_arr)
max_price = np.max(valid_prices_arr)

print("\n--- Question 2: Car Price Statistics (NumPy) ---")
print(f"Average Price: ${avg_price:,.2f}")
print(f"Minimum Price: ${min_price:,.2f}")
print(f"Maximum Price: ${max_price:,.2f}")

# ---- Cell 3 ----
# Question 1: Car manufacturers that appear most often
make_counts = df["make"].value_counts()
print("\n--- Question 1: Top Car Manufacturers by Count ---")
print(make_counts)

# Question 3: Body style with the highest average price
body_avg_price = (
    df.dropna(subset=["price"])
    .groupby("body-style")["price"]
    .mean()
    .sort_values(ascending=False)
)
print("\n--- Question 3: Average Price by Body Style ---")
print(body_avg_price)
print(
    f"Highest average price body style: {body_avg_price.index[0]}"
    f" (${body_avg_price.iloc[0]:,.2f})"
)

# Question 4: Compare average engine size across fuel types
fuel_engine_stats = (
    df.groupby("fuel-type")["engine-size"]
    .agg(count="count", avg_engine_size="mean", median_engine_size="median")
    .sort_values(by="avg_engine_size", ascending=False)
)
print("\n--- Question 4: Engine Size Across Fuel Types ---")
print(fuel_engine_stats)

# Question 6: 10 most expensive cars (using filtering + sorting)
priced_cars = df[df["price"].notna()]
top_10_expensive = priced_cars.sort_values(by="price", ascending=False).head(10)[
    ["make", "body-style", "fuel-type", "engine-size", "horsepower", "price"]
]
print("\n--- Question 6: 10 Most Expensive Cars ---")
print(top_10_expensive.to_string(index=False))

# ---- Cell 4 ----
# Chart 1: Manufacturer-count bar chart
plt.figure(figsize=(10, 5))
make_counts.plot(kind="bar", color="steelblue", edgecolor="black")
plt.title("Number of Cars by Manufacturer (UCI Automobile Dataset)")
plt.xlabel("Manufacturer (make)")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("manufacturer_counts.png", dpi=150)
plt.show()

# Chart 2: Price histogram
plt.figure(figsize=(8, 5))
plt.hist(
    valid_prices_arr, bins=15, color="teal", edgecolor="black", alpha=0.85
)
plt.title("Distribution of Car Prices")
plt.xlabel("Price (USD)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("price_histogram.png", dpi=150)
plt.show()

