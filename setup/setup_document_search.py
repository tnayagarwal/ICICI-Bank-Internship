#!/usr/bin/env python3
"""
Document Search Setup System
- Extracts text from PDF documents
- Stores in PostgreSQL with full-text search
- Creates hybrid search capabilities
- Optimized for HR policy document search
"""
import sys
import os
import re
import hashlib
from pathlib import Path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.postgres_client import postgres_client

# Try to import PDF processing libraries
USE_PDFPLUMBER = False
try:
    import pdfplumber
    PDF_AVAILABLE = True
    USE_PDFPLUMBER = True
except ImportError:
    try:
        import PyPDF2
        PDF_AVAILABLE = True
        USE_PDFPLUMBER = False
    except ImportError:
        PDF_AVAILABLE = False
        USE_PDFPLUMBER = False

def create_documents_table():
    """Create documents table with full-text search capabilities"""
    print("🗄️  Creating documents table...")
    
    with postgres_client._get_cursor() as cur:
        # Drop existing table if it exists
        cur.execute("DROP TABLE IF EXISTS policy_documents")
        
        # Create new table with full-text search support
        cur.execute("""
            CREATE TABLE policy_documents (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                title VARCHAR(500) NOT NULL,
                content TEXT NOT NULL,
                content_hash VARCHAR(64) NOT NULL UNIQUE,
                document_type VARCHAR(100),
                keywords TEXT,
                page_count INTEGER,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                search_vector tsvector
            )
        """)
        
        # Create full-text search index
        cur.execute("""
            CREATE INDEX idx_documents_search 
            ON policy_documents 
            USING GIN(search_vector)
        """)
        
        # Create additional indexes for faster queries
        cur.execute("CREATE INDEX idx_documents_filename ON policy_documents(filename)")
        cur.execute("CREATE INDEX idx_documents_type ON policy_documents(document_type)")
        cur.execute("CREATE INDEX idx_documents_title ON policy_documents(title)")
        
        print("  ✅ Documents table created with search indexes")

def extract_text_from_pdf(file_path):
    """Extract text from PDF using available libraries"""
    try:
        text = ""
        page_count = 0
        
        if USE_PDFPLUMBER:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        
        return text.strip(), page_count
    
    except Exception as e:
        print(f"  ⚠️  Error extracting from {file_path}: {str(e)}")
        return "", 0

def clean_and_process_text(text):
    """Clean and process extracted text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'\n\d+\n', '\n', text)
    text = re.sub(r'Page \d+ of \d+', '', text)
    
    # Fix common PDF extraction issues
    text = text.replace('Ã¢â‚¬â„¢', "'")
    text = text.replace('Ã¢â‚¬Å"', '"')
    text = text.replace('Ã¢â‚¬', '"')
    
    return text.strip()

def extract_keywords(text, filename):
    """Extract relevant keywords from text and filename"""
    keywords = []
    
    # Common HR policy keywords
    hr_keywords = [
        'policy', 'procedure', 'employee', 'employer', 'workplace', 'conduct',
        'ethics', 'harassment', 'discrimination', 'leave', 'vacation', 'sick',
        'maternity', 'paternity', 'grievance', 'complaint', 'disciplinary',
        'termination', 'resignation', 'promotion', 'performance', 'appraisal',
        'benefits', 'compensation', 'salary', 'wages', 'overtime', 'bonus',
        'provident fund', 'gratuity', 'insurance', 'medical', 'safety',
        'security', 'confidentiality', 'whistleblower', 'code of conduct',
        'equal opportunity', 'diversity', 'inclusion', 'training', 'development'
    ]
    
    # Extract keywords from filename
    filename_words = re.findall(r'[a-zA-Z]+', filename.lower())
    keywords.extend(filename_words)
    
    # Find HR keywords in text
    text_lower = text.lower()
    for keyword in hr_keywords:
        if keyword in text_lower:
            keywords.append(keyword)
    
    # Extract potential policy names (capitalized phrases)
    policy_names = re.findall(r'[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', text)
    keywords.extend([name.lower() for name in policy_names[:10]])  # Limit to first 10
    
    return ', '.join(set(keywords))

def determine_document_type(filename, text):
    """Determine document type based on filename and content"""
    filename_lower = filename.lower()
    text_lower = text.lower()
    
    # Define document type patterns
    type_patterns = {
        'Code of Conduct': ['code', 'conduct', 'ethics', 'business ethics'],
        'Harassment Policy': ['harassment', 'sexual harassment', 'workplace harassment'],
        'Leave Policy': ['leave', 'vacation', 'maternity', 'paternity', 'sick leave'],
        'Employment Law': ['employment', 'labor', 'labour', 'industrial disputes', 'factories act'],
        'Compensation Policy': ['wages', 'salary', 'bonus', 'gratuity', 'provident fund', 'epf'],
        'Safety Policy': ['safety', 'security', 'workplace safety'],
        'Whistleblower Policy': ['whistleblower', 'complaint', 'grievance'],
        'Banking Regulation': ['banking', 'rbi', 'reserve bank', 'icici'],
        'ESG Report': ['esg', 'sustainability', 'environmental'],
        'General Policy': ['policy', 'procedure', 'guidelines']
    }
    
    # Check patterns
    for doc_type, patterns in type_patterns.items():
        for pattern in patterns:
            if pattern in filename_lower or pattern in text_lower:
                return doc_type
    
    return 'General Document'

def generate_title(filename, text):
    """Generate a meaningful title for the document"""
    # Clean filename
    title = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    title = re.sub(r'\d+', '', title).strip()
    
    # Look for title in first few lines of text
    lines = text.split('\n')[:10]
    for line in lines:
        line = line.strip()
        if len(line) > 10 and len(line) < 100 and line.isupper():
            return line.title()
        elif len(line) > 10 and len(line) < 100 and any(word in line.lower() for word in ['policy', 'act', 'code', 'guidelines']):
            return line.strip()
    
    # Fallback to cleaned filename
    return title.title()

def process_documents():
    """Process all PDF documents in the data folder"""
    print("📄 Processing PDF documents...")
    
    if not PDF_AVAILABLE:
        print("❌ PDF processing libraries not available. Please install PyPDF2 or pdfplumber:")
        print("   pip install PyPDF2")
        print("   or")
        print("   pip install pdfplumber")
        return False
    
    data_folder = Path("data")
    if not data_folder.exists():
        print("❌ Data folder not found")
        return False
    
    pdf_files = list(data_folder.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in data folder")
        return False
    
    print(f"📋 Found {len(pdf_files)} PDF files")
    processed_count = 0
    
    with postgres_client._get_cursor() as cur:
        for pdf_file in pdf_files:
            try:
                print(f"  🔄 Processing {pdf_file.name}...")
                
                # Extract text
                text, page_count = extract_text_from_pdf(pdf_file)
                if not text:
                    print(f"    ⚠️  No text extracted from {pdf_file.name}")
                    continue
                
                # Process text
                cleaned_text = clean_and_process_text(text)
                if len(cleaned_text) < 100:
                    print(f"    ⚠️  Insufficient text content in {pdf_file.name}")
                    continue
                
                # Generate metadata
                content_hash = hashlib.sha256(cleaned_text.encode()).hexdigest()
                title = generate_title(pdf_file.name, cleaned_text)
                doc_type = determine_document_type(pdf_file.name, cleaned_text)
                keywords = extract_keywords(cleaned_text, pdf_file.name)
                file_size = pdf_file.stat().st_size
                
                # Check if document already exists
                cur.execute("SELECT id FROM policy_documents WHERE content_hash = %s", (content_hash,))
                if cur.fetchone():
                    print(f"    ⏭️  {pdf_file.name} already processed (same content)")
                    continue
                
                # Insert document
                cur.execute("""
                    INSERT INTO policy_documents 
                    (filename, title, content, content_hash, document_type, keywords, page_count, file_size, search_vector)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, to_tsvector('english', %s || ' ' || %s || ' ' || %s))
                """, (
                    pdf_file.name,
                    title,
                    cleaned_text,
                    content_hash,
                    doc_type,
                    keywords,
                    page_count,
                    file_size,
                    title,
                    keywords,
                    cleaned_text[:5000]  # Limit for search vector
                ))
                
                processed_count += 1
                print(f"    ✅ Processed {pdf_file.name} ({len(cleaned_text)} chars, {page_count} pages)")
                
            except Exception as e:
                print(f"    ❌ Error processing {pdf_file.name}: {str(e)}")
                continue
    
    print(f"  ✅ Successfully processed {processed_count} documents")
    return processed_count > 0

def create_search_functions():
    """Create PostgreSQL functions for advanced search"""
    print("🔍 Creating search functions...")
    
    with postgres_client._get_cursor() as cur:
        # Drop existing function if it exists
        cur.execute("DROP FUNCTION IF EXISTS search_policy_documents(TEXT, INTEGER)")
        
        # Create function for hybrid search
        cur.execute("""
            CREATE OR REPLACE FUNCTION search_policy_documents(
                search_query TEXT,
                limit_count INTEGER DEFAULT 5
            )
            RETURNS TABLE(
                doc_id INTEGER,
                filename VARCHAR,
                title VARCHAR,
                document_type VARCHAR,
                relevance_score DOUBLE PRECISION,
                content_snippet TEXT
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    pd.id,
                    pd.filename,
                    pd.title,
                    pd.document_type,
                    (
                        ts_rank(pd.search_vector, plainto_tsquery('english', search_query)) * 2 +
                        CASE 
                            WHEN pd.title ILIKE '%' || search_query || '%' THEN 0.5
                            WHEN pd.keywords ILIKE '%' || search_query || '%' THEN 0.3
                            ELSE 0
                        END
                    ) as relevance_score,
                    LEFT(pd.content, 500) as content_snippet
                FROM policy_documents pd
                WHERE 
                    pd.search_vector @@ plainto_tsquery('english', search_query)
                    OR pd.title ILIKE '%' || search_query || '%'
                    OR pd.keywords ILIKE '%' || search_query || '%'
                    OR pd.content ILIKE '%' || search_query || '%'
                ORDER BY relevance_score DESC
                LIMIT limit_count;
            END;
            $$ LANGUAGE plpgsql;
        """)
        
        print("  ✅ Search functions created")

def verify_setup():
    """Verify the document search setup"""
    print("\n✅ Verifying document search setup...")
    
    with postgres_client._get_cursor() as cur:
        # Check document count
        cur.execute("SELECT COUNT(*) as count FROM policy_documents")
        doc_count = cur.fetchone()['count']
        print(f"📊 Total documents: {doc_count}")
        
        # Check document types
        cur.execute("SELECT document_type, COUNT(*) as count FROM policy_documents GROUP BY document_type ORDER BY count DESC")
        doc_types = cur.fetchall()
        print("📋 Document types:")
        for doc_type in doc_types:
            print(f"  {doc_type['document_type']}: {doc_type['count']} documents")
        
        # Test search function
        print("\n🔍 Testing search function with 'leave policy'...")
        cur.execute("SELECT * FROM search_policy_documents('leave policy', 3)")
        results = cur.fetchall()
        if results:
            print("  ✅ Search function working:")
            for result in results:
                print(f"    {result['title']} (Score: {result['relevance_score']:.2f})")
        else:
            print("  ⚠️  No results found for test query")

def main():
    """Main setup function"""
    print("🗄️  Document Search System Setup")
    print("=" * 50)
    
    try:
        # Test database connection
        with postgres_client._get_cursor() as cur:
            cur.execute("SELECT 1")
        print("✅ Database connection successful")
        
        # Create tables and indexes
        create_documents_table()
        
        # Process documents
        if process_documents():
            # Create search functions
            create_search_functions()
            
            # Verify setup
            verify_setup()
            
            print("\n" + "=" * 50)
            print("🎉 Document search system setup completed successfully!")
            return True
        else:
            print("❌ Document processing failed")
            return False
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 