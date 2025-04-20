import sqlite3
import os
import streamlit as st

def store_dataframe_per_ticker_to_sqlite(df, db_name="market_data.db"):
    if os.path.exists(db_name):
        print(f"Database '{db_name}' already exists. Skipping creation.")
        return
    
    conn = sqlite3.connect(db_name)
    tickers = df['Ticker'].unique()
    
    for ticker in tickers:
        df_ticker = df[df['Ticker'] == ticker].reset_index(drop=True)
        table_name = ticker.replace('.', '_').replace('-', '_')
        df_ticker.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Stored data for {ticker} into SQLite table: {table_name}")
    
    conn.close()
    print(f"Database '{db_name}' created successfully.")


def append_data_to_ticker(ticker_name, new_data_df, db_name="market_data.db"):
    table_name = ticker_name.replace('.', '_').replace('-', '_')
    
    conn = sqlite3.connect(db_name)
    
    new_data_df.to_sql(table_name, conn, if_exists='append', index=False)
    
    conn.close()
    print(f"Successfully added {len(new_data_df)} rows to table '{table_name}'.")

def update_columns_in_db(ticker_name, date, data, db_name="market_data.db"):
    table_name = ticker_name.replace('.', '_').replace('-', '_')
    
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE Date = ?", (date,))
    exists = cursor.fetchone()[0] > 0

    if exists:
        update_values = []
        update_clause = []
        
        for col, value in data.items():
            if value is not None: 
                update_clause.append(f"{col} = ?")
                update_values.append(value)
        
        if update_clause:
            update_values.append(date)
            update_stmt = f"UPDATE {table_name} SET {', '.join(update_clause)} WHERE Date = ?"
            
            cursor.execute(update_stmt, update_values)
            conn.commit()
            print(f"Updated columns for {ticker_name} on {date}.")
        else:
            print(f"No valid data to update for {ticker_name} on {date}.")
    else:
        append_data_to_ticker(ticker_name, data)

    conn.close()
