import time
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Data
labels = ['CSV', 'Parquet', 'Excel', 'SQL', 'Pickle']
times = [6.68, 2.93, 212.28, 10.66, 0.03]

# Create bar graph for saving times
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, times, color=['blue', 'green', 'red', 'purple', 'orange'])

# Add titles and labels
plt.title('Iteration 1 Data Saving Times')
plt.xlabel('Data Format')
plt.ylabel('Time (s)')

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.show()

# Measure time to read CSV
start_time = time.time()
df_csv = pd.read_csv('dataframe_1_redundancy.csv')
csv_time = time.time() - start_time

# Measure time to read Parquet
start_time = time.time()
df_parquet = pd.read_parquet('dataframe_1_redundancy.parquet')
parquet_time = time.time() - start_time

# Measure time to read Excel
start_time = time.time()
df_excel = pd.read_excel('dataframe_1_redundancy.xlsx')
excel_time = time.time() - start_time

# Measure time to read SQL
start_time = time.time()
conn = sqlite3.connect('dataframe_1_redundancy.db')
df_sql = pd.read_sql_query('SELECT * FROM data', conn)
sql_time = time.time() - start_time
conn.close()

# Measure time to read Pickle
start_time = time.time()
df_pickle = pd.read_pickle('dataframe_1_redundancy.pkl')
pickle_time = time.time() - start_time

# Print the times
print(f"CSV read time: {csv_time:.2f} seconds")
print(f"Parquet read time: {parquet_time:.2f} seconds")
print(f"Excel read time: {excel_time:.2f} seconds")
print(f"SQL read time: {sql_time:.2f} seconds")
print(f"Pickle read time: {pickle_time:.2f} seconds")

# Data for reading times
read_times = [csv_time, parquet_time, excel_time, sql_time, pickle_time]

# Create bar graph for reading times
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, read_times, color=['blue', 'green', 'red', 'purple', 'orange'])

# Add titles and labels
plt.title('Iteration 1 Data Reading Times')
plt.xlabel('Data Format')
plt.ylabel('Time (s)')

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

# Show the plot
plt.show()