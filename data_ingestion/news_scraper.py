"""
News Scraper Module
Fetches latest financial news using NewsAPI and parses content with BeautifulSoup.
"""

import requests
import json
from bs4 import BeautifulSoup
from config import NEWS_API_KEY
from typing import List, Dict


def fetch_financial_news(query: str = "finance", language: str = "en", 
                        sort_by: str = "publishedAt", page_size: int = 20) -> List[Dict]:
    """
    Fetch financial news from NewsAPI.
    
    Args:
        query: Search query for news (default: "finance")
        language: Language code (default: "en")
        sort_by: Sort order (default: "publishedAt")
        page_size: Number of articles to fetch (default: 20)
    
    Returns:
        List of news articles with metadata
    """
    url = "https://newsapi.org/v2/everything"
    
    params = {
        'q': query,
        'language': language,
        'sortBy': sort_by,
        'pageSize': page_size,
        'apiKey': NEWS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get('articles', [])
        
        # Process articles to extract clean content
        processed_articles = []
        for article in articles:
            processed_article = {
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'publishedAt': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', ''),
                'content': extract_article_content(article.get('url', ''))
            }
            processed_articles.append(processed_article)
            
        return processed_articles
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []


def extract_article_content(url: str) -> str:
    """
    Extract main content from article URL using BeautifulSoup.
    
    Args:
        url: Article URL
    
    Returns:
        Extracted text content
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Try to find main content areas
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            'main',
            '.content'
        ]
        
        content = ""
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text() for elem in elements])
                break
        
        # Fallback to body if no specific content found
        if not content:
            content = soup.get_text()
        
        # Clean up whitespace
        content = ' '.join(content.split())
        
        return content[:2000]  # Limit content length
        
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return ""


def get_market_news() -> List[Dict]:
    """
    Get general market news.
    
    Returns:
        List of market news articles
    """
    return fetch_financial_news(query="stock market OR earnings OR financial results")


def get_company_news(company: str) -> List[Dict]:
    """
    Get news for a specific company.
    
    Args:
        company: Company name or ticker symbol
    
    Returns:
        List of company-specific news articles
    """
    return fetch_financial_news(query=f"{company} earnings OR {company} financial")


def main():
    """Sample usage of the news scraper."""
    print("Fetching latest financial news...")
    
    # Get general market news
    market_news = get_market_news()
    print(f"Found {len(market_news)} market news articles")
    
    # Get Tesla-specific news
    tesla_news = get_company_news("Tesla")
    print(f"Found {len(tesla_news)} Tesla news articles")
    
    # Display sample article
    if market_news:
        sample = market_news[0]
        print(f"\nSample Article:")
        print(f"Title: {sample['title']}")
        print(f"Source: {sample['source']}")
        print(f"Published: {sample['publishedAt']}")
        print(f"Content preview: {sample['content'][:200]}...")


if __name__ == "__main__":
    main()