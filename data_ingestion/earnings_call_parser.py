"""
Earnings Call Parser Module
Parses earnings call transcripts and extracts structured information.
"""

import re
from typing import Dict, List


def parse_earnings_transcript(transcript_text: str) -> Dict:
    """
    Parse earnings call transcript and extract structured information.
    
    Args:
        transcript_text: Raw transcript text
    
    Returns:
        Dictionary containing parsed transcript data
    """
    # Clean the transcript
    cleaned_text = clean_transcript_text(transcript_text)
    
    # Extract basic information
    parsed_data = {
        'company': extract_company_name(cleaned_text),
        'quarter': extract_quarter_info(cleaned_text),
        'date': extract_call_date(cleaned_text),
        'participants': extract_participants(cleaned_text),
        'sections': extract_transcript_sections(cleaned_text),
        'raw_text': cleaned_text
    }
    
    return parsed_data


def clean_transcript_text(text: str) -> str:
    """
    Clean and normalize transcript text.
    
    Args:
        text: Raw transcript text
    
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common transcript artifacts
    artifacts = [
        r'\[.*?\]',  # [Operator] [Unidentified Analyst]
        r'\(.*?\)',  # (Operator) (Unidentified Analyst)
        r'Operator:',
        r'Thank you\.',
        r'Next question\.',
    ]
    
    for pattern in artifacts:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()


def extract_company_name(text: str) -> str:
    """
    Extract company name from transcript.
    
    Args:
        text: Transcript text
    
    Returns:
        Company name
    """
    # Look for common patterns
    patterns = [
        r'Welcome to (.+?) earnings call',
        r'(.+?) Q[1-4] \d{4} earnings call',
        r'(.+?) earnings conference call',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return "Unknown Company"


def extract_quarter_info(text: str) -> str:
    """
    Extract quarter and year information.
    
    Args:
        text: Transcript text
    
    Returns:
        Quarter information (e.g., "Q1 2024")
    """
    patterns = [
        r'Q[1-4] \d{4}',
        r'(\d{1,2})(?:st|nd|rd|th) quarter \d{4}',
        r'(\d{4}) (Q[1-4])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    return "Unknown Quarter"


def extract_call_date(text: str) -> str:
    """
    Extract call date from transcript.
    
    Args:
        text: Transcript text
    
    Returns:
        Call date
    """
    # Look for date patterns
    date_patterns = [
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}/\d{1,2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0).strip()
    
    return "Unknown Date"


def extract_participants(text: str) -> List[str]:
    """
    Extract list of participants from transcript.
    
    Args:
        text: Transcript text
    
    Returns:
        List of participant names
    """
    participants = []
    
    # Look for speaker patterns
    speaker_patterns = [
        r'([A-Z][a-z]+ [A-Z][a-z]+):',  # First Last:
        r'([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+):',  # First Middle Last:
    ]
    
    for pattern in speaker_patterns:
        matches = re.findall(pattern, text)
        participants.extend(matches)
    
    # Remove duplicates and common non-speakers
    participants = list(set(participants))
    participants = [p for p in participants if p not in ['Operator', 'Thank you', 'Next question']]
    
    return participants[:10]  # Limit to top 10 participants


def extract_transcript_sections(text: str) -> Dict[str, str]:
    """
    Extract different sections of the transcript.
    
    Args:
        text: Transcript text
    
    Returns:
        Dictionary with section names and content
    """
    sections = {}
    
    # Split into potential sections
    section_markers = [
        'opening remarks',
        'prepared remarks',
        'q&a',
        'question and answer',
        'closing remarks',
        'conclusion'
    ]
    
    text_lower = text.lower()
    
    for marker in section_markers:
        if marker in text_lower:
            # Find the section content
            pattern = rf'.*?{re.escape(marker)}.*?(?=\n\n|\n[A-Z][a-z]+:|$)'
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[marker.replace(' ', '_')] = match.group(0).strip()
    
    return sections


def chunk_transcript(transcript_data: Dict, chunk_size: int = 1000) -> List[Dict]:
    """
    Split transcript into smaller chunks for processing.
    
    Args:
        transcript_data: Parsed transcript data
        chunk_size: Maximum chunk size in characters
    
    Returns:
        List of transcript chunks with metadata
    """
    chunks = []
    text = transcript_data['raw_text']
    
    # Split by sentences first
    sentences = re.split(r'[.!?]+', text)
    
    current_chunk = ""
    chunk_id = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed chunk size, save current chunk
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append({
                'chunk_id': chunk_id,
                'text': current_chunk.strip(),
                'company': transcript_data['company'],
                'quarter': transcript_data['quarter'],
                'date': transcript_data['date'],
                'chunk_type': 'transcript'
            })
            chunk_id += 1
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add the last chunk
    if current_chunk:
        chunks.append({
            'chunk_id': chunk_id,
            'text': current_chunk.strip(),
            'company': transcript_data['company'],
            'quarter': transcript_data['quarter'],
            'date': transcript_data['date'],
            'chunk_type': 'transcript'
        })
    
    return chunks


def main():
    """Sample usage of the earnings call parser."""
    # Sample transcript text
    sample_transcript = """
    Welcome to Apple Inc. Q1 2024 earnings call. Today's call is being recorded.
    
    Participants:
    Tim Cook - CEO
    Luca Maestri - CFO
    Kate Adams - General Counsel
    
    Tim Cook: Thank you for joining us today. We're pleased to report strong results for Q1 2024...
    
    Luca Maestri: Revenue for the quarter was $123.9 billion, up 8% year over year...
    
    Q&A Session:
    Operator: Thank you. We'll now begin the question and answer session.
    
    Analyst: What's your outlook for iPhone sales in China?
    Tim Cook: We remain optimistic about our position in China...
    """
    
    print("Parsing sample earnings call transcript...")
    
    # Parse the transcript
    parsed_data = parse_earnings_transcript(sample_transcript)
    
    print(f"Company: {parsed_data['company']}")
    print(f"Quarter: {parsed_data['quarter']}")
    print(f"Date: {parsed_data['date']}")
    print(f"Participants: {parsed_data['participants']}")
    print(f"Sections found: {list(parsed_data['sections'].keys())}")
    
    # Create chunks
    chunks = chunk_transcript(parsed_data, chunk_size=500)
    print(f"\nCreated {len(chunks)} chunks")
    
    # Display first chunk
    if chunks:
        print(f"\nFirst chunk preview:")
        print(f"Text: {chunks[0]['text'][:200]}...")


if __name__ == "__main__":
    main()