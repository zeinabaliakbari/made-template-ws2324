import pandas as pd
import sqlite3
from io import StringIO
import requests

def fetch_csv_data(url):
    response = requests.get(url)
    data = response.text
    return "\n".join(data.splitlines()[6:-4])

def read_and_clean_csv(data):
    df = pd.read_csv(StringIO(data), encoding='utf-8', sep=";")
    new_column_names = ['date', 'CIN', 'name', 'petrol', 'diesel', 'gas', 'electro', 'hybrid', 'plugInHybrid', 'others']
    df.columns = range(len(df.columns))
    df = df[[0, 1, 2, 12, 22, 32, 42, 52, 62, 72]]
    df.columns = new_column_names

    df['CIN'] = df['CIN'].astype(str).apply(lambda x: f"{int(x):05}" if x.isdigit() else x)
    numeric_columns = ['petrol', 'diesel', 'gas', 'electro', 'hybrid', 'plugInHybrid', 'others']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[(df[col].notnull()) & (df[col] > 0)]
    return df

def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False, encoding='utf-8')

def read_csv(file_path):
    return pd.read_csv(file_path)

def print_first_row(df):
    print("First row of the DataFrame:")
    print(df.head(1))

def connect_to_database(db_name):
    return sqlite3.connect(db_name)

def create_sqlite_table(conn, df, table_name, sqlite_types):
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join([f'{col} {sqlite_types[col]}' for col in df.columns])})"
    conn.execute(create_table_query)

def write_to_sqlite(df, conn, table_name, sqlite_types):
    df.to_sql(table_name, conn, index=False, if_exists="replace", dtype=sqlite_types)

def close_connection(conn):
    conn.close()
    print("Connection closed.")

# Step 1: Fetch the CSV data from the URL
url = "https://www-genesis.destatis.de/genesis/downloads/00/tables/46251-0021_00.csv"
csv_data = fetch_csv_data(url)

# Step 2: Read and clean the CSV data
df = read_and_clean_csv(csv_data)

# Step 3: Save cleaned data to a new CSV file
##new_csv_path = 'cleaned_data.csv'
##save_to_csv(df, new_csv_path)

# Step 4: Read the new CSV file
##df_new = read_csv(new_csv_path)

# Step 5: Print the first row of the new CSV file
##print_first_row(df_new)

# Step 6: Connect to SQLite database
db_name = "cars.sqlite"
conn = connect_to_database(db_name)

# Step 7: Create SQLite table with appropriate types
sqlite_types = {'date': 'TEXT', 'CIN': 'TEXT', 'name': 'TEXT',
                'petrol': 'BIGINT', 'diesel': 'BIGINT', 'gas': 'BIGINT',
                'electro': 'BIGINT', 'hybrid': 'BIGINT', 'plugInHybrid': 'BIGINT', 'others': 'BIGINT'}
table_name = 'cars'
create_sqlite_table(conn, df, table_name, sqlite_types)

# Step 8: Write DataFrame to SQLite database
write_to_sqlite(df, conn, table_name, sqlite_types)

# Step 9: Close the connection
close_connection(conn)
