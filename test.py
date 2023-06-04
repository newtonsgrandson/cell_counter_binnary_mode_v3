import pandas as pd

# Create a sample DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
})

print("Original DataFrame:")
print(df)

# Add 10 to all values in the DataFrame
df = df + 10

print("\nDataFrame after adding 10 to all values:")
print(df)