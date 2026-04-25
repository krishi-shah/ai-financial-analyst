"""
Financial Reports Parser Module
Parses PDF financial reports (10-K, 10-Q, etc.) using PyMuPDF.
"""

import fitz  # PyMuPDF
import re
from typing import Dict, List
from pathlib import Path


def parse_pdf_report(pdf_path: str) -> Dict:
    """
    Parse a PDF financial report and extract structured information.
    
    Args:
        pdf_path: Path to the PDF file
    
    Returns:
        Dictionary containing parsed report data
    """
    try:
        # Open PDF document
        doc = fitz.open(pdf_path)
        
        # Extract metadata
        metadata = doc.metadata
        
        # Extract text from all pages
        full_text = ""
        page_texts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            page_text = page.get_text()
            page_texts.append({
                'page_number': page_num + 1,
                'text': page_text
            })
            full_text += page_text + "\n"
        
        total_pages = len(doc)
        doc.close()
        
        parsed_data = {
            'file_path': pdf_path,
            'metadata': metadata,
            'total_pages': total_pages,
            'full_text': full_text,
            'page_texts': page_texts,
            'sections': extract_report_sections(full_text),
            'financial_data': extract_financial_data(full_text),
            'risk_factors': extract_risk_factors(full_text)
        }
        
        return parsed_data
        
    except Exception as e:
        print(f"Error parsing PDF {pdf_path}: {e}")
        return {}


def extract_report_sections(text: str) -> Dict[str, str]:
    """
    Extract different sections from financial report.
    
    Args:
        text: Full report text
    
    Returns:
        Dictionary with section names and content
    """
    sections = {}
    
    # Common section headers in financial reports
    section_headers = [
        'Item 1. Business',
        'Item 1A. Risk Factors',
        'Item 2. Properties',
        'Item 3. Legal Proceedings',
        'Item 4. Mine Safety Disclosures',
        'Item 5. Market for Registrant',
        'Item 6. Selected Financial Data',
        'Item 7. Management',
        'Item 7A. Quantitative and Qualitative Disclosures',
        'Item 8. Financial Statements',
        'Item 9. Changes in and Disagreements',
        'Item 9A. Controls and Procedures',
        'Item 9B. Other Information',
        'Item 10. Directors, Executive Officers',
        'Item 11. Executive Compensation',
        'Item 12. Security Ownership',
        'Item 13. Certain Relationships',
        'Item 14. Principal Accountant Fees',
        'Item 15. Exhibits and Financial Statement Schedules'
    ]
    
    for header in section_headers:
        # Look for the section header
        pattern = rf'{re.escape(header)}.*?(?=\n\s*Item \d+\.|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            section_content = match.group(0).strip()
            # Clean up the section name for dictionary key
            section_key = header.replace(' ', '_').replace('.', '').lower()
            sections[section_key] = section_content
    
    return sections


def extract_financial_data(text: str) -> Dict[str, List[str]]:
    """
    Extract financial data and metrics from the report.
    
    Args:
        text: Report text
    
    Returns:
        Dictionary with financial data categories
    """
    financial_data = {
        'revenue': [],
        'net_income': [],
        'assets': [],
        'liabilities': [],
        'cash_flow': [],
        'ratios': []
    }
    
    # Revenue patterns
    revenue_patterns = [
        r'revenue[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
        r'total revenue[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
        r'net sales[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?'
    ]
    
    for pattern in revenue_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        financial_data['revenue'].extend(matches)
    
    # Net income patterns
    income_patterns = [
        r'net income[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
        r'net earnings[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
        r'profit[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?'
    ]
    
    for pattern in income_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        financial_data['net_income'].extend(matches)
    
    # Asset patterns
    asset_patterns = [
        r'total assets[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?',
        r'current assets[:\s]*\$?([0-9,]+(?:\.[0-9]+)?)\s*(?:million|billion|thousand)?'
    ]
    
    for pattern in asset_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        financial_data['assets'].extend(matches)
    
    return financial_data


def extract_risk_factors(text: str) -> List[str]:
    """
    Extract risk factors from the report.
    
    Args:
        text: Report text
    
    Returns:
        List of risk factor descriptions
    """
    risk_factors = []
    
    # Look for risk factors section
    risk_section_pattern = r'Item 1A\.?\s*Risk Factors.*?(?=Item \d+\.|$)'
    risk_match = re.search(risk_section_pattern, text, re.IGNORECASE | re.DOTALL)
    
    if risk_match:
        risk_section = risk_match.group(0)
        
        # Extract individual risk factors
        risk_patterns = [
            r'•\s*([^•]+?)(?=•|$)',
            r'\d+\.\s*([^0-9]+?)(?=\d+\.|$)',
            r'-\s*([^-]+?)(?=-|$)'
        ]
        
        for pattern in risk_patterns:
            matches = re.findall(pattern, risk_section, re.IGNORECASE | re.DOTALL)
            risk_factors.extend([match.strip() for match in matches if len(match.strip()) > 20])
    
    return risk_factors[:20]  # Limit to top 20 risk factors


def chunk_report(report_data: Dict, chunk_size: int = 1000) -> List[Dict]:
    """
    Split report into smaller chunks for processing.
    
    Args:
        report_data: Parsed report data
        chunk_size: Maximum chunk size in characters
    
    Returns:
        List of report chunks with metadata
    """
    chunks = []
    text = report_data['full_text']
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    chunk_id = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed chunk size, save current chunk
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'file_path': report_data['file_path'],
                'chunk_type': 'financial_report',
                'metadata': report_data['metadata']
            })
            chunk_id += 1
            current_chunk = paragraph
        else:
            current_chunk += "\n\n" + paragraph if current_chunk else paragraph
    
    # Add the last chunk
    if current_chunk:
        chunks.append({
            'chunk_id': chunk_id,
            'text': current_chunk.strip(),
            'file_path': report_data['file_path'],
            'chunk_type': 'financial_report',
            'metadata': report_data['metadata']
        })
    
    return chunks


def parse_multiple_reports(directory_path: str) -> List[Dict]:
    """
    Parse multiple PDF reports from a directory.
    
    Args:
        directory_path: Path to directory containing PDF files
    
    Returns:
        List of parsed report data
    """
    directory = Path(directory_path)
    pdf_files = list(directory.glob("*.pdf"))
    
    parsed_reports = []
    
    for pdf_file in pdf_files:
        print(f"Parsing {pdf_file.name}...")
        report_data = parse_pdf_report(str(pdf_file))
        if report_data:
            parsed_reports.append(report_data)
    
    return parsed_reports


def main():
    """Sample usage of the reports parser."""
    print("Financial Reports Parser")
    print("This module parses PDF financial reports and extracts structured data.")
    print("\nTo use this module:")
    print("1. Place PDF files in a directory")
    print("2. Call parse_pdf_report('path/to/file.pdf') for single file")
    print("3. Call parse_multiple_reports('path/to/directory') for multiple files")
    
    # Example of how to use the parser
    sample_usage = """
    # Parse a single PDF
    report_data = parse_pdf_report('path/to/10k_report.pdf')
    
    # Extract sections
    sections = report_data['sections']
    print(f"Found {len(sections)} sections")
    
    # Extract financial data
    financial_data = report_data['financial_data']
    print(f"Revenue mentions: {financial_data['revenue']}")
    
    # Create chunks for processing
    chunks = chunk_report(report_data)
    print(f"Created {len(chunks)} chunks")
    """
    
    print("\nSample usage:")
    print(sample_usage)


if __name__ == "__main__":
    main()