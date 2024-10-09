import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyspark
from pyspark.sql import SparkSession
import numpy as np
import time
import os 

os.environ['PYSPARK_PYTHON'] = 
os.environ['PYSPARK_DRIVER_PYTHON'] = 

# Create a large dataset
def generate_data(n_rows=10**6):
    data = {
        'id': np.arange(n_rows),
        'value1': np.random.randn(n_rows),
        'value2': np.random.randint(0, 100, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], size=n_rows)
    }
    return pd.DataFrame(data)

# Function to benchmark performance
def benchmark_pandas(df):
    start_time = time.time()
    
    # Simple filtering and groupby operation
    result = df[df['value2'] > 50].groupby('category').agg({'value1': 'mean'})
    
    elapsed_time = time.time() - start_time
    print(f"Regular Pandas took {elapsed_time:.4f} seconds")
    # free memory
    del result
    return elapsed_time

def benchmark_pandas_pyarrow(df):
    # Convert to Arrow Table (this might speed up some operations, depending on usage)
    arrow_table = pa.Table.from_pandas(df)
    
    start_time = time.time()
    
    # Convert back to pandas and perform similar operations
    df_arrow = arrow_table.to_pandas()
    result = df_arrow[df_arrow['value2'] > 50].groupby('category').agg({'value1': 'mean'})
    
    elapsed_time = time.time() - start_time
    print(f"Pandas with PyArrow took {elapsed_time:.4f} seconds")
    # free memory
    del result
    return elapsed_time

def benchmark_pyspark(spark, df):
    # Convert pandas DataFrame to Spark DataFrame
    
    spark_df = spark.createDataFrame(df)
    
    # Perform similar operations using PySpark
    result = spark_df.filter(spark_df['value2'] > 50)\
                     .groupBy('category')\
                     .agg({'value1': 'mean'})
    
    start_time = time.time()
    
    result.collect()  # Force computation
    elapsed_time = time.time() - start_time
    print(f"PySpark took {elapsed_time:.4f} seconds")
    # free memory
    del result
    return elapsed_time

# Initialize Spark session
spark = SparkSession.builder.appName('PerformanceComparison').getOrCreate()


# Generate dataset
df = generate_data(10**5)

# Run benchmarks
print("Running performance comparison on 1 million rows...\n")

time_pandas = benchmark_pandas(df)
time_pandas_pyarrow = benchmark_pandas_pyarrow(df)
time_pyspark = benchmark_pyspark(spark, df)

# Final comparison
print(f"\nPerformance Summary:\n")
print(f"Regular Pandas: {time_pandas:.4f} seconds")
print(f"Pandas with PyArrow: {time_pandas_pyarrow:.4f} seconds")
print(f"PySpark: {time_pyspark:.4f} seconds")

# Stop Spark session
spark.stop()

import matplotlib.pyplot as plt

# Plot the results
labels = ['Pandas with PyArrow','Pandas', 'PySpark']
times = [time_pandas_pyarrow,time_pandas,  time_pyspark]

plt.figure(figsize=(10, 6))
plt.bar(labels, times, color=['blue', 'orange', 'green'])
# Add values on the plot
for i, v in enumerate(times):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom')
plt.xlabel('Method')
plt.ylabel('Time (seconds)')
plt.title('Performance Comparison for a groupby operation over 100K rows')
plt.show()
