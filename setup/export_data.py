#!/usr/bin/env python3
"""
Export Data Script
Automatically exports all PostgreSQL table data to CSV files
"""
import sys
import os
import csv
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.postgres_client import postgres_client

def ensure_export_directory():
    """Ensure the export directory exists"""
    export_dir = "data"
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)
        print(f"📁 Created directory: {export_dir}")
    else:
        print(f"📁 Using existing directory: {export_dir}")
    return export_dir

def get_table_schema(table_name):
    """Get detailed schema information for a table"""
    try:
        with postgres_client._get_cursor() as cur:
            # Get column information
            cur.execute("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable,
                    column_default,
                    ordinal_position
                FROM information_schema.columns 
                WHERE table_name = %s 
                ORDER BY ordinal_position
            """, (table_name,))
            columns = cur.fetchall()
            
            # Get row count
            cur.execute(f"SELECT COUNT(*) as count FROM {table_name}")
            count_result = cur.fetchone()
            row_count = count_result['count'] if count_result else 0
            
            # Check if table has an 'id' column
            has_id_column = any(col['column_name'] == 'id' for col in columns)
            
            return {
                'columns': columns,
                'row_count': row_count,
                'has_id_column': has_id_column,
                'column_names': [col['column_name'] for col in columns]
            }
            
    except Exception as e:
        print(f"❌ Error getting schema for {table_name}: {str(e)}")
        return None

def export_table_to_csv(table_name, schema_info, export_dir):
    """Export a table to CSV file"""
    try:
        print(f"🔄 Exporting {table_name} ({schema_info['row_count']} records)...")
        
        if schema_info['row_count'] == 0:
            print(f"⏭️  Skipping {table_name} - no data")
            return None
        
        # Build query based on available columns
        column_names = schema_info['column_names']
        columns_str = ", ".join(column_names)
        
        # Use ORDER BY id if available, otherwise just select all
        if schema_info['has_id_column']:
            query = f"SELECT {columns_str} FROM {table_name} ORDER BY id"
        else:
            query = f"SELECT {columns_str} FROM {table_name}"
        
        # Execute query
        with postgres_client._get_cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
        
        if not results:
            print(f"⚠️  No data returned from {table_name}")
            return None
        
        # Create CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"{table_name}_export_{timestamp}.csv"
        csv_filepath = os.path.join(export_dir, csv_filename)
        
        # Write to CSV
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=column_names)
            
            # Write header
            writer.writeheader()
            
            # Write data
            for record in results:
                # Convert complex data types to strings for CSV
                row_data = {}
                for key, value in record.items():
                    if isinstance(value, (dict, list)):
                        row_data[key] = json.dumps(value, default=str)
                    elif value is None:
                        row_data[key] = ""
                    else:
                        row_data[key] = str(value)
                
                writer.writerow(row_data)
        
        print(f"✅ Exported {table_name}: {csv_filepath}")
        return csv_filepath
        
    except Exception as e:
        print(f"❌ Failed to export {table_name}: {str(e)}")
        return None

def main():
    """Main export function"""
    print("🗄️ PostgreSQL Database CSV Export Tool")
    print("=" * 60)
    
    # Test database connection
    try:
        with postgres_client._get_cursor() as cur:
            cur.execute("SELECT 1")
        print("✅ Database connection successful")
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False
    
    # Ensure export directory exists
    export_dir = ensure_export_directory()
    
    # Get all tables
    print("\n🔍 Discovering database tables...")
    try:
        with postgres_client._get_cursor() as cur:
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name
            """)
            tables = cur.fetchall()
            table_names = [table['table_name'] for table in tables]
    except Exception as e:
        print(f"❌ Failed to get table list: {str(e)}")
        return False
    
    if not table_names:
        print("⚠️  No tables found in database")
        return False
    
    print(f"📋 Found {len(table_names)} tables: {', '.join(table_names)}")
    
    # Analyze table schemas
    print("\n🔍 Analyzing table schemas...")
    table_schemas = {}
    total_rows = 0
    
    for table_name in table_names:
        print(f"   Checking {table_name}...", end="")
        schema = get_table_schema(table_name)
        
        if schema:
            table_schemas[table_name] = schema
            total_rows += schema['row_count']
            status = f" {schema['row_count']} rows, {len(schema['columns'])} columns"
            print(status)
        else:
            print(" ERROR")
    
    print(f"\n📊 Total rows across all tables: {total_rows}")
    
    if total_rows == 0:
        print("⚠️  No data found in any table")
        return False
    
    # Export all tables with data
    print(f"\n🚀 Starting automatic export to CSV files...")
    print(f"📁 Export directory: {os.path.abspath(export_dir)}")
    
    exported_files = []
    tables_with_data = [name for name, schema in table_schemas.items() if schema['row_count'] > 0]
    
    print(f"📋 Tables to export: {', '.join(tables_with_data)}")
    print("-" * 60)
    
    for table_name in tables_with_data:
        schema = table_schemas[table_name]
        csv_file = export_table_to_csv(table_name, schema, export_dir)
        
        if csv_file:
            exported_files.append(csv_file)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Export completed!")
    print(f"📊 Exported {len(exported_files)} tables")
    print(f"📁 Files saved in: {os.path.abspath(export_dir)}")
    
    if exported_files:
        print("\n📄 Exported files:")
        for file in exported_files:
            filename = os.path.basename(file)
            filesize = os.path.getsize(file)
            print(f"   {filename} ({filesize:,} bytes)")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print(f"\n✅ All exports completed successfully!")
        else:
            print(f"\n❌ Export failed!")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 