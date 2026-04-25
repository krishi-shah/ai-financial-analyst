"""
Local LLM Module
Provides alternative to OpenAI API using local models or simple text generation.
"""

import re
from typing import Dict, List


class LocalFinancialLLM:
    """
    Local LLM alternative for generating financial responses.
    """
    
    def __init__(self):
        """Initialize the local LLM."""
        self.financial_templates = {
            'revenue': [
                "Based on the financial data, {company} reported revenue of {amount} in {period}.",
                "The company's revenue for {period} was {amount}, showing {trend} compared to previous periods.",
                "Revenue figures indicate {company} generated {amount} in {period}."
            ],
            'earnings': [
                "The earnings report shows {company} achieved {amount} in {period}.",
                "Financial performance indicates {company} earned {amount} during {period}.",
                "The company's earnings for {period} totaled {amount}."
            ],
            'growth': [
                "Growth metrics show {company} experienced {trend} in {period}.",
                "The company demonstrated {trend} growth during {period}.",
                "Performance indicators suggest {trend} growth for {company} in {period}."
            ]
        }
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate answer using local templates and context analysis.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Generated answer
        """
        # Analyze query type
        query_type = self._analyze_query_type(query)
        
        # Extract key information from context
        extracted_info = self._extract_financial_info(context)
        
        # Generate structured response
        answer = self._generate_structured_response(query, context, query_type, extracted_info)
        
        return answer
    
    def _analyze_query_type(self, query: str) -> str:
        """
        Analyze query to determine the type of financial information requested.
        
        Args:
            query: User query
        
        Returns:
            Query type (revenue, earnings, growth, etc.)
        """
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['revenue', 'sales', 'income']):
            return 'revenue'
        elif any(word in query_lower for word in ['earnings', 'profit', 'net income']):
            return 'earnings'
        elif any(word in query_lower for word in ['growth', 'increase', 'decrease', 'change']):
            return 'growth'
        elif any(word in query_lower for word in ['delivery', 'deliver', 'vehicles', 'units']):
            return 'delivery'
        else:
            return 'general'
    
    def _extract_financial_info(self, context: str) -> Dict:
        """
        Extract financial information from context.
        
        Args:
            context: Retrieved context
        
        Returns:
            Dictionary with extracted financial information
        """
        info = {
            'company': self._extract_company(context),
            'amount': self._extract_amounts(context),
            'period': self._extract_period(context),
            'trend': self._extract_trend(context)
        }
        
        return info
    
    def _extract_company(self, context: str) -> str:
        """Extract company name from context."""
        companies = ['Apple', 'Tesla', 'Microsoft', 'Amazon', 'Google', 'Meta', 'Netflix']
        
        for company in companies:
            if company.lower() in context.lower():
                return company
        
        return "the company"
    
    def _extract_amounts(self, context: str) -> List[str]:
        """Extract monetary amounts from context."""
        # Pattern for monetary amounts
        amount_pattern = r'\$[\d,]+\.?\d*\s*(?:billion|million|thousand)?'
        amounts = re.findall(amount_pattern, context, re.IGNORECASE)
        
        # Also look for numbers with units
        number_pattern = r'[\d,]+\.?\d*\s*(?:billion|million|thousand|%|percent)'
        numbers = re.findall(number_pattern, context, re.IGNORECASE)
        
        return amounts + numbers
    
    def _extract_period(self, context: str) -> str:
        """Extract time period from context."""
        periods = ['Q1', 'Q2', 'Q3', 'Q4', 'quarter', 'year', '2023', '2024']
        
        for period in periods:
            if period.lower() in context.lower():
                return period
        
        return "the reported period"
    
    def _extract_trend(self, context: str) -> str:
        """Extract trend information from context."""
        if any(word in context.lower() for word in ['up', 'increase', 'growth', 'rise', 'surge']):
            return "positive"
        elif any(word in context.lower() for word in ['down', 'decrease', 'decline', 'fall', 'drop']):
            return "negative"
        else:
            return "stable"
    
    def _generate_structured_response(self, query: str, context: str, 
                                    query_type: str, extracted_info: Dict) -> str:
        """
        Generate structured response based on query type and extracted information.
        
        Args:
            query: Original query
            context: Retrieved context
            query_type: Type of query
            extracted_info: Extracted financial information
        
        Returns:
            Structured response
        """
        response_parts = []
        
        # Start with direct answer
        response_parts.append("**Answer:**")
        
        # Generate specific response based on query type
        if query_type == 'revenue' and extracted_info['amount']:
            amount = extracted_info['amount'][0] if extracted_info['amount'] else "reported amount"
            company = extracted_info['company']
            period = extracted_info['period']
            
            response_parts.append(f"According to the financial data, {company} reported revenue of {amount} in {period}.")
        
        elif query_type == 'earnings' and extracted_info['amount']:
            amount = extracted_info['amount'][0] if extracted_info['amount'] else "reported earnings"
            company = extracted_info['company']
            period = extracted_info['period']
            
            response_parts.append(f"The earnings report shows {company} achieved {amount} in {period}.")
        
        elif query_type == 'delivery':
            # Look for delivery/vehicle numbers
            delivery_pattern = r'[\d,]+\.?\d*\s*(?:vehicles|units|deliveries)'
            deliveries = re.findall(delivery_pattern, context, re.IGNORECASE)
            
            if deliveries:
                response_parts.append(f"The company delivered {deliveries[0]} as reported in the data.")
            else:
                response_parts.append("Delivery information was found in the financial data, but specific numbers weren't clearly extracted.")
        
        else:
            # General response
            response_parts.append("Based on the available financial data, here's what I found:")
        
        # Add key findings
        response_parts.append("\n**Key Findings:**")
        
        if extracted_info['amount']:
            response_parts.append(f"• Financial figures: {', '.join(extracted_info['amount'][:3])}")
        
        if extracted_info['trend'] != 'stable':
            response_parts.append(f"• Trend: {extracted_info['trend']} performance")
        
        # Add source information
        response_parts.append(f"\n**Source:** {extracted_info['company']} {extracted_info['period']} financial data")
        
        # Add note about local processing
        response_parts.append("\n💡 **Note**: This response was generated using local AI processing. For more sophisticated analysis, consider using OpenAI API when quota is available.")
        
        return "\n".join(response_parts)


def main():
    """Test the local LLM."""
    llm = LocalFinancialLLM()
    
    sample_query = "What was Apple's revenue in Q4 2023?"
    sample_context = """
    [Source 1] Company: Apple | Quarter: Q4 2023 | Source: earnings_call
    Apple reported Q4 2023 revenue of $94.8 billion, up 1% year-over-year, driven by strong iPhone sales.
    
    [Source 2] Company: Apple | Quarter: Q4 2023 | Source: earnings_call
    The company's services revenue grew significantly during the quarter.
    """
    
    answer = llm.generate_answer(sample_query, sample_context)
    print("Sample Query:", sample_query)
    print("\nGenerated Answer:")
    print(answer)
    print("\n" + "="*50)


if __name__ == "__main__":
    main()
