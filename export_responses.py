import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('responses.db')  # Make sure this is in the same directory

# Query to select all data
query = "SELECT * FROM responses"

# Load data into pandas DataFrame
df = pd.read_sql_query(query, conn)

# Export DataFrame to CSV
df.to_csv('responses_export.csv', index=False)

# Close the database connection
conn.close()

print("✅ Data exported successfully to 'responses_export.csv'")
