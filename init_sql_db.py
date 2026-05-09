import sqlite3
import os

def init_db():
    db_path = 'company_data.db'
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            salary REAL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            employee_id INTEGER,
            amount REAL,
            date TEXT,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    ''')
    
    # Insert dummy data
    employees_data = [
        (1, 'Alice Smith', 'Engineering', 120000),
        (2, 'Bob Johnson', 'Sales', 85000),
        (3, 'Charlie Brown', 'Sales', 90000),
        (4, 'Diana Prince', 'Management', 150000)
    ]
    cursor.executemany('INSERT INTO employees VALUES (?,?,?,?)', employees_data)
    
    sales_data = [
        (1, 2, 5000, '2023-01-15'),
        (2, 2, 7500, '2023-02-20'),
        (3, 3, 6200, '2023-01-18'),
        (4, 3, 8900, '2023-03-05')
    ]
    cursor.executemany('INSERT INTO sales VALUES (?,?,?,?)', sales_data)
    
    conn.commit()
    conn.close()
    print(f"✅ Successfully created and populated {db_path} with dummy data.")

if __name__ == "__main__":
    init_db()
