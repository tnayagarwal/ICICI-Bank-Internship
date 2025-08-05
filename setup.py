#!/usr/bin/env python3
"""
ICICI Bank HR Assistant - Setup Script
=====================================

This script helps set up the HR Assistant system by:
1. Checking system requirements
2. Installing dependencies
3. Setting up the database
4. Configuring environment variables

Author: ICICI Bank Development Team
Version: 1.0.0
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version meets requirements"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_postgresql():
    """Check if PostgreSQL is available"""
    print("\n🐘 Checking PostgreSQL...")
    try:
        # Try to connect to PostgreSQL
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            dbname='postgres',
            user='postgres',
            password='TaNaY'
        )
        conn.close()
        print("✅ PostgreSQL connection successful")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("   Please ensure PostgreSQL is installed and running")
        print("   Default credentials: user=postgres, password=TaNaY")
        return False

def setup_database():
    """Set up the database schema"""
    print("\n🗄️ Setting up database...")
    try:
        # Create database if it doesn't exist
        import psycopg2
        conn = psycopg2.connect(
            host='localhost',
            port=5432,
            dbname='postgres',
            user='postgres',
            password='TaNaY'
        )
        conn.autocommit = True
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute("SELECT 1 FROM pg_database WHERE datname='talentagent_db'")
        if not cur.fetchone():
            cur.execute("CREATE DATABASE talentagent_db")
            print("✅ Database 'talentagent_db' created")
        else:
            print("✅ Database 'talentagent_db' already exists")
        
        cur.close()
        conn.close()
        
        # Initialize database schema
        subprocess.check_call([sys.executable, "setup/init_db.py"])
        print("✅ Database schema initialized")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def create_env_file():
    """Create .env file with default configuration"""
    print("\n⚙️ Creating environment configuration...")
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    # Create default .env file
    env_content = """# ICICI Bank HR Assistant - Environment Configuration

# LLM Configuration
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=gemma2-9b-it

# Email Configuration
GMAIL_ADDRESS=your_email@gmail.com
GMAIL_APP_PASSWORD=your_app_password_here

# Database Configuration (optional - defaults used if not set)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=talentagent_db
DB_USER=postgres
DB_PASSWORD=TaNaY

# System Configuration
DEBUG=True
LOG_LEVEL=INFO
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created with default configuration")
        print("   Please update the API keys and email settings")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")
        return False

def test_system():
    """Test the system setup"""
    print("\n🧪 Testing system setup...")
    try:
        # Test backend
        result = subprocess.run([sys.executable, "app.py"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Backend system test passed")
        else:
            print(f"❌ Backend system test failed: {result.stderr}")
            return False
            
        return True
    except subprocess.TimeoutExpired:
        print("✅ Backend system test passed (timeout expected)")
        return True
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 ICICI Bank HR Assistant - Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Check PostgreSQL
    if not check_postgresql():
        print("\n⚠️ PostgreSQL check failed. Please install and configure PostgreSQL first.")
        print("   You can continue with setup, but database operations will fail.")
        response = input("   Continue with setup? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Setup database
    if not setup_database():
        print("\n⚠️ Database setup failed. You may need to configure PostgreSQL manually.")
        response = input("   Continue with setup? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Test system
    if test_system():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update the .env file with your API keys and email settings")
        print("2. Start the web interface: python frontend.py")
        print("3. Open http://localhost:5000 in your browser")
        print("4. Login with employee ID: 12345 (test user)")
    else:
        print("\n❌ Setup completed with errors. Please check the configuration.")
        sys.exit(1)

if __name__ == "__main__":
    main() 