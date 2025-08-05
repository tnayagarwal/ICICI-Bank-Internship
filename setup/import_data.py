#!/usr/bin/env python3
'''
Database Import Script
Imports CSV data into PostgreSQL database
'''

import sys
import os
import csv
import psycopg2
import psycopg2.extras
import json
from datetime import datetime

# Database configuration - UPDATE THESE VALUES
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'talentagent_db',
    'user': 'postgres',
    'password': 'TaNaY'  # UPDATED PASSWORD
}

def get_database_connection():
    '''Get database connection'''
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return None

def import_csv_to_table(table_name, csv_file):
    '''Import CSV data to table with proper data type handling'''
    conn = get_database_connection()
    if not conn:
        return False
    
    try:
        # Increase field size limit for large content
        csv.field_size_limit(1000000)  # 1MB limit
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader)
            
            # Clear existing data
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {table_name}")
                
                # Special handling for different tables
                if table_name == 'agent_feedback':
                    # Handle agent_response as JSONB
                    for row in reader:
                        processed_row = []
                        for i, cell in enumerate(row):
                            if headers[i] == 'agent_response' and cell:
                                # Try to parse as JSON, if fails store as string
                                try:
                                    json.loads(cell)
                                    processed_row.append(cell)
                                except:
                                    processed_row.append(json.dumps(cell))
                            elif headers[i] == 'improvement_applied':
                                # Convert to boolean
                                processed_row.append(cell.lower() in ['true', '1', 'yes'] if cell else False)
                            elif headers[i] == 'rating':
                                # Convert to integer
                                processed_row.append(int(cell) if cell and cell.isdigit() else None)
                            else:
                                processed_row.append(None if cell == '' or cell == 'null' else cell)
                        
                        placeholders = ', '.join(['%s'] * len(headers))
                        insert_sql = f"INSERT INTO {table_name} ({', '.join(headers)}) VALUES ({placeholders})"
                        cur.execute(insert_sql, processed_row)
                
                elif table_name == 'attendance':
                    # Handle date and time conversions
                    for row in reader:
                        processed_row = []
                        for i, cell in enumerate(row):
                            if headers[i] == 'date' and cell:
                                try:
                                    # Try different date formats
                                    if '/' in cell:
                                        processed_row.append(datetime.strptime(cell, '%m/%d/%Y').date())
                                    else:
                                        processed_row.append(datetime.strptime(cell, '%Y-%m-%d').date())
                                except:
                                    processed_row.append(None)
                            elif headers[i] in ['entry', 'exit'] and cell and cell != 'null':
                                try:
                                    processed_row.append(datetime.strptime(cell, '%H:%M:%S').time())
                                except:
                                    processed_row.append(None)
                            else:
                                processed_row.append(None if cell == '' or cell == 'null' else cell)
                        
                placeholders = ', '.join(['%s'] * len(headers))
                insert_sql = f"INSERT INTO {table_name} ({', '.join(headers)}) VALUES ({placeholders})"
                        cur.execute(insert_sql, processed_row)
                
                else:
                    # Standard import for other tables
                rows_inserted = 0
                for row in reader:
                    # Convert empty strings to None for NULL values
                        processed_row = [None if cell == '' or cell == 'null' else cell for cell in row]
                        
                        placeholders = ', '.join(['%s'] * len(headers))
                        insert_sql = f"INSERT INTO {table_name} ({', '.join(headers)}) VALUES ({placeholders})"
                    cur.execute(insert_sql, processed_row)
                    rows_inserted += 1
                
                conn.commit()
                print(f"✅ Imported {table_name}: imported successfully")
                return True
                
    except Exception as e:
        print(f"❌ Error importing {table_name}: {str(e)}")
        return False
    finally:
        conn.close()

def main():
    '''Main import function'''
    print("📥 Database CSV Import Tool")
    print("=" * 50)
    
    # Test database connection
    conn = get_database_connection()
    if not conn:
        print("❌ Cannot connect to database. Please check your DB_CONFIG.")
        return False
    conn.close()
    
    csv_dir = 'csv_data'
    if not os.path.exists(csv_dir):
        print(f"❌ CSV directory '{csv_dir}' not found")
        return False
    
    # Import tables in correct order (respecting foreign keys)
    tables_order = [
        'employees',
        'attendance', 
        'leaves',
        'leave_requests',
        'leave_balances',
        'complaints',
        'policy_documents',
        'data',
        'agent_feedback'
    ]
    
    success_count = 0
    total_count = 0
    
    for table_name in tables_order:
        csv_file = os.path.join(csv_dir, f"{table_name}.csv")
        if os.path.exists(csv_file):
            print(f"\n📊 Importing {table_name}...")
            if import_csv_to_table(table_name, csv_file):
                success_count += 1
            total_count += 1
        else:
            print(f"⚠️  CSV file not found for table: {table_name}")
    
    print(f"\n📊 Import Summary: {success_count}/{total_count} tables imported successfully")
    
    if success_count == total_count:
        print("✅ All data imported successfully!")
        return True
    else:
        print("⚠️  Some imports failed. Check the logs above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
