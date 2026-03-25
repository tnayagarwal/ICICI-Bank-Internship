import psycopg2
import csv
import os
from datetime import datetime

DB_CONFIG = dict(
    host='localhost',
    port=5432,
    dbname='talentagent_db',
    user='postgres',
    password='TaNaY',
)

def drop_all_tables():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    # Drop all tables if they exist, in order to avoid FK issues
    tables = [
        'agent_feedback', 'data', 'policy_documents', 'complaints', 'leave_balances', 'leave_requests', 'leaves', 'attendance', 'employees'
    ]
    for table in tables:
        try:
            cur.execute(f'DROP TABLE IF EXISTS {table} CASCADE;')
        except Exception as e:
            print(f"Error dropping table {table}: {e}")
    cur.close()
    conn.close()
    print('All tables dropped successfully.')

def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    
    # Create employees table (matches employees.csv)
    cur.execute('''
        CREATE TABLE employees (
            id VARCHAR(20) PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100),
            leaves_available INT DEFAULT 20,
            last_leave DATE,
            dept_head VARCHAR(100),
            head_email VARCHAR(100),
            auto_approved_leaves JSONB DEFAULT '[]'::jsonb,
            approval_pending_leaves JSONB DEFAULT '[]'::jsonb
        );
    ''')
    
    # Create attendance table (matches attendance.csv)
    cur.execute('''
        CREATE TABLE attendance (
            srno SERIAL PRIMARY KEY,
            emp_id VARCHAR(20),
            date DATE,
            entry TIME,
            exit TIME,
            status VARCHAR(50) DEFAULT 'normal',
            override_reason TEXT,
            created_at TIMESTAMP DEFAULT now(),
            FOREIGN KEY (emp_id) REFERENCES employees(id)
        );
    ''')
    
    # Create leaves table (matches leaves.csv structure)
    cur.execute('''
        CREATE TABLE leaves (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            leaves_left INT,
            last_leave TEXT,
            last_ai_approved TEXT,
            pending_approval_hr TEXT
        );
    ''')
    
    # Create leave_requests table (matches leave_requests.csv exactly)
    cur.execute('''
        CREATE TABLE leave_requests (
            id SERIAL PRIMARY KEY,
            emp_id VARCHAR(20),
            leave_type VARCHAR(50),
            start_date DATE,
            end_date DATE,
            days_requested DECIMAL(3,1),
            reason TEXT,
            status VARCHAR(20) DEFAULT 'pending',
            applied_date TIMESTAMP,
            approved_by VARCHAR(100),
            approved_date TIMESTAMP,
            FOREIGN KEY (emp_id) REFERENCES employees(id)
        );
    ''')
    
    # Create leave_balances table (matches leave_balances.csv exactly)
    cur.execute('''
        CREATE TABLE leave_balances (
            emp_id VARCHAR(20),
            annual_leave DECIMAL(4,1),
            sick_leave DECIMAL(4,1),
            casual_leave DECIMAL(4,1),
            maternity_leave DECIMAL(4,1),
            paternity_leave DECIMAL(4,1),
            last_updated TIMESTAMP,
            FOREIGN KEY (emp_id) REFERENCES employees(id)
        );
    ''')
    
    # Create complaints table (matches complaints.csv)
    cur.execute('''
        CREATE TABLE complaints (
            id SERIAL PRIMARY KEY,
            emp_id VARCHAR(20),
            subject TEXT,
            content TEXT,
            status VARCHAR(20) DEFAULT 'open',
            created_at TIMESTAMP DEFAULT now(),
            resolved_at TIMESTAMP,
            FOREIGN KEY (emp_id) REFERENCES employees(id)
        );
    ''')
    
    # Create policy_documents table (matches policy_documents.csv exactly)
    cur.execute('''
        CREATE TABLE policy_documents (
            id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            title VARCHAR(255),
            content TEXT,
            content_hash VARCHAR(255),
            document_type VARCHAR(100),
            keywords TEXT,
            page_count INT,
            file_size BIGINT,
            created_at TIMESTAMP DEFAULT now(),
            search_vector tsvector
        );
    ''')
    
    # Create data table (matches data.csv)
    cur.execute('''
        CREATE TABLE data (
            id SERIAL PRIMARY KEY,
            content TEXT,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT now()
        );
    ''')
    
    # Create agent_feedback table (matches agent_feedback.csv with all columns)
    cur.execute('''
        CREATE TABLE agent_feedback (
            id SERIAL PRIMARY KEY,
            agent_type VARCHAR(50),
            emp_id VARCHAR(50),
            original_request TEXT,
            agent_response JSONB,
            user_feedback TEXT,
            improvement_applied BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            rating INT,
            comments TEXT
        );
    ''')
    
    cur.close()
    conn.close()
    print('All tables created successfully.')

def import_csv_data():
    """Import data from CSV files"""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    
    # Import employee data from second CSV
    employee_csv_path = "Supabase Snippet List Non-Template Databases (1).csv"
    if os.path.exists(employee_csv_path):
        with open(employee_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    last_leave = datetime.strptime(row['last leave'], '%Y-%m-%d').date() if row['last leave'] else None
                    cur.execute('''
                        INSERT INTO employees (id, name, email, leaves_available, last_leave, dept_head, head_email)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name,
                        email = EXCLUDED.email,
                        leaves_available = EXCLUDED.leaves_available,
                        last_leave = EXCLUDED.last_leave,
                        dept_head = EXCLUDED.dept_head,
                        head_email = EXCLUDED.head_email
                    ''', (row['id'], row['name'], row['email'], int(row['leaves']), 
                         last_leave, row['dept head'], row['head email']))
                except Exception as e:
                    print(f"Error importing employee {row.get('id', 'unknown')}: {e}")
        print(f"Imported employee data from {employee_csv_path}")
    
    # Import attendance data from first CSV
    attendance_csv_path = "Supabase Snippet List Non-Template Databases.csv"
    if os.path.exists(attendance_csv_path):
        with open(attendance_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    date_obj = datetime.strptime(row['date'], '%Y-%m-%d').date()
                    entry_time = datetime.strptime(row['entry'], '%H:%M:%S').time() if row['entry'] else None
                    exit_time = datetime.strptime(row['exit'], '%H:%M:%S').time() if row['exit'] and row['exit'] != 'null' else None
                    
                    cur.execute('''
                        INSERT INTO attendance (emp_id, date, entry, exit)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                    ''', (row['id'], date_obj, entry_time, exit_time))
                except Exception as e:
                    print(f"Error importing attendance for {row.get('id', 'unknown')}: {e}")
        print(f"Imported attendance data from {attendance_csv_path}")
    
    cur.close()
    conn.close()
    print('CSV data imported successfully.')

if __name__ == '__main__':
    drop_all_tables()
    init_db()
    import_csv_data() 