#!/usr/bin/env python3
"""
Fixes the attendance table's primary key sequence (srno) in PostgreSQL.
Run this after importing attendance data or moving the database to a new machine.
If the sequence is not found, prints the SQL command to run manually.
"""

import sys
from utils.postgres_client import postgres_client
from loguru import logger

def fix_attendance_sequence():
    try:
        with postgres_client._get_cursor() as cur:
            # Find the max srno in attendance
            cur.execute("SELECT MAX(srno) FROM attendance;")
            max_srno = cur.fetchone()[0] or 0
            # Try to get the sequence name
            cur.execute("SELECT pg_get_serial_sequence('attendance', 'srno');")
            seq_name = cur.fetchone()[0]
            if not seq_name:
                print("❌ Could not find sequence for attendance.srno.")
                print("If your table was created without SERIAL or IDENTITY, you must create a sequence and set it manually.")
                print(f"To fix manually, run in psql (replace sequence_name if needed):\n\n  SELECT setval('attendance_srno_seq', {max_srno + 1}, false);")
                return
            cur.execute(f"SELECT setval('{seq_name}', {max_srno + 1}, false);")
            print(f"✅ Attendance sequence '{seq_name}' set to {max_srno + 1}")
    except Exception as e:
        logger.error(f"Failed to fix attendance sequence: {e}")
        print(f"❌ Error: {e}")
        print("If this is a sequence/serial issue, see the README for manual fix instructions.")

def main():
    print("🔧 Fixing attendance table sequence...")
    fix_attendance_sequence()
    print("Done.")

if __name__ == "__main__":
    main() 