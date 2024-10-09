import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import pickle
from concurrent.futures import ThreadPoolExecutor

# Function to measure file size
def file_size_in_mb(file_path):
    return os.path.getsize(file_path) / (1024 * 1024)  # in MB

# Function to save files and measure sizes (runs in parallel)
def save_and_measure(df, iteration, redundancy_ratio):
    # Save DataFrame in different formats
    csv_file = f"dataframe_{iteration+1}_redundancy.csv"
    parquet_file = f"dataframe_{iteration+1}_redundancy.parquet"
    excel_file = f"dataframe_{iteration+1}_redundancy.xlsx"
    sql_file = f"dataframe_{iteration+1}_redundancy.db"
    pickle_file = f"dataframe_{iteration+1}_redundancy.pkl"

    df.to_csv(csv_file, index=False)
    df.to_parquet(parquet_file)
    df.to_excel(excel_file, index=False)

    # Save DataFrame to SQLite database
    conn = sqlite3.connect(sql_file)
    df.to_sql('data', conn, if_exists='replace', index=False)
    conn.close()

    # Save DataFrame as a pickle file
    with open(pickle_file, 'wb') as f:
        pickle.dump(df, f)

    # Measure file sizes
    csv_size = file_size_in_mb(csv_file)
    parquet_size = file_size_in_mb(parquet_file)
    excel_size = file_size_in_mb(excel_file)
    sql_size = file_size_in_mb(sql_file)
    pickle_size = file_size_in_mb(pickle_file)

    print(f"Iteration {iteration + 1} | Redundancy: {redundancy_ratio:.2f} | "
          f"CSV Size: {csv_size:.2f} MB | Parquet Size: {parquet_size:.2f} MB | "
          f"Excel Size: {excel_size:.2f} MB | SQL Size: {sql_size:.2f} MB | "
          f"Pickle Size: {pickle_size:.2f} MB")
    
    return csv_size, parquet_size, excel_size, sql_size, pickle_size

# Initialize parameters
rows = 10_000  # Number of rows
cols = 100  # Number of columns
iterations = 10  # Number of iterations for redundancy
redundancy_ratios = np.linspace(0, 1, iterations)  # Redundancy ratios from 0 to 1

# Lists to store file sizes per iteration
csv_sizes = []
parquet_sizes = []
excel_sizes = []
sql_sizes = []
pickle_sizes = []

# Multithreading: Create a ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    futures = []
    
    for i, redundancy_ratio in enumerate(redundancy_ratios):
        print(f"Preparing iteration {i + 1} with redundancy ratio {redundancy_ratio:.2f}")
        
        # Generate DataFrame with progressive redundancy
        df = pd.DataFrame(np.random.randn(rows, cols), columns=[f'col_{i}' for i in range(cols)])
        # Add a column with a random succession of letters
        df['random_letters'] = [''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=10)) for _ in range(rows)]

        # Put the random letters in the first column to introduce redundancy
        df.iloc[:, 0] = df['random_letters']

        # Introduce redundancy progressively
        redundant_cols = int(cols * redundancy_ratio)
        if redundant_cols > 0:
            df.iloc[:, -redundant_cols:] = df.iloc[:, :redundant_cols].values  # Duplicate data in the last few columns
        
        # Submit the task to the thread pool for parallel execution
        futures.append(executor.submit(save_and_measure, df, i, redundancy_ratio))

    # Collect results after all threads finish
    for future in futures:
        csv_size, parquet_size, excel_size, sql_size, pickle_size = future.result()
        csv_sizes.append(csv_size)
        parquet_sizes.append(parquet_size)
        excel_sizes.append(excel_size)
        sql_sizes.append(sql_size)
        pickle_sizes.append(pickle_size)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(redundancy_ratios, csv_sizes, label='CSV', marker='o')
plt.plot(redundancy_ratios, parquet_sizes, label='Parquet', marker='o')
plt.plot(redundancy_ratios, excel_sizes, label='Excel', marker='o')
plt.plot(redundancy_ratios, sql_sizes, label='SQL', marker='o')
plt.plot(redundancy_ratios, pickle_sizes, label='Pickle (Ram usage)', marker='o')
plt.xlabel('Redundancy Ratio')
plt.ylabel('File Size (MB)')
plt.title('File Size vs Redundancy in Data (Multithreaded)')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()
