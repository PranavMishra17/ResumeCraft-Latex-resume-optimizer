import re
import logging
from typing import List

logger = logging.getLogger(__name__)

def extract_name_from_resume(latex_content: str) -> str:
    """Extract name from LaTeX resume for PDF naming"""
    # Look for common name patterns
    patterns = [
        r'\\name\{([^}]+)\}',
        r'\\textbf\{\\huge\s+([^}]+)\}',
        r'\\begin\{center\}\\textbf\{\\Large\s+([^}]+)\}',
        r'\\textbf\{\\Large\s+([^}]+)\}',
        r'\\huge\{([^}]+)\}',
        r'\\Large\{([^}]+)\}'
    ]

    for pattern in patterns:
        match = re.search(pattern, latex_content, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Clean LaTeX commands from name
            name = re.sub(r'\\[a-zA-Z]+\s*', '', name).strip()
            if name and len(name.split()) <= 4:  # Reasonable name length
                return name

    # Fallback: look for first capitalized words in document
    lines = latex_content.split('\n')
    for line in lines:
        if '\\begin{document}' in line:
            continue
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', line)
        if words and len(words[0].split()) <= 4:
            return words[0]

    return "Resume"

def escape_latex_keywords(keywords: List[str]) -> List[str]:
    """Escape special LaTeX characters in keywords"""
    escaped = []
    for keyword in keywords:
        # Escape ampersand and other special characters
        escaped_keyword = keyword.replace('&', '\\&')
        escaped_keyword = escaped_keyword.replace('%', '\\%')
        escaped_keyword = escaped_keyword.replace('$', '\\$')
        escaped_keyword = escaped_keyword.replace('#', '\\#')
        escaped.append(escaped_keyword)
    return escaped

def count_keyword_occurrences(text: str, keyword: str) -> int:
    """Count actual keyword occurrences, avoiding false positives"""
    # Clean keyword for comparison (remove LaTeX escaping)
    clean_keyword = keyword.replace('\\&', '&').replace('\\%', '%').replace('\\$', '$').replace('\\#', '#')
    
    # Handle special cases
    if clean_keyword.lower() == 'r':
        # For 'R', only count when it appears as a standalone word or in specific contexts
        import re
        # Count R in programming contexts, not in LaTeX commands
        pattern = r'\b[Rr]\b(?!\s*[a-z])'  # R not followed by lowercase (to avoid LaTeX commands)
        matches = re.findall(pattern, text)
        # Additional context-based filtering
        valid_matches = 0
        for match in re.finditer(pattern, text):
            context = text[max(0, match.start()-20):match.end()+20].lower()
            # Count if it's in programming context
            if any(prog_term in context for prog_term in ['programming', 'language', 'statistical', 'analysis', 'data']):
                valid_matches += 1
        return valid_matches
    
    elif clean_keyword.lower() == 'c#':
        # For C#, look for explicit mentions
        import re
        pattern = r'\b[Cc]#\b|\b[Cc]-?[Ss]harp\b'
        return len(re.findall(pattern, text))
    
    elif len(clean_keyword) <= 2:
        # For other short keywords, use word boundary matching
        import re
        pattern = r'\b' + re.escape(clean_keyword) + r'\b'
        return len(re.findall(pattern, text, re.IGNORECASE))
    
    else:
        # For longer keywords, use case-insensitive substring matching
        return text.lower().count(clean_keyword.lower())

def count_words_strict(text: str) -> int:
    """Count words strictly excluding LaTeX commands"""
    # Remove all LaTeX commands and their arguments
    clean_text = re.sub(r'\\[a-zA-Z*]+(\[[^\]]*\])?(\{[^}]*\})*', '', text)
    # Remove URLs
    clean_text = re.sub(r'https?://[^\s}]+', '', clean_text)
    # Remove special characters and braces
    clean_text = re.sub(r'[{}\\&%]', '', clean_text)
    # Count actual words
    words = [w for w in clean_text.split() if w.strip()
             and not w.isdigit()]
    return len(words)

def count_characters_strict(text: str) -> int:
    """Count characters excluding LaTeX commands for strict character limits"""
    # Remove all LaTeX commands and their arguments
    clean_text = re.sub(r'\\[a-zA-Z*]+(\[[^\]]*\])?(\{[^}]*\})*', '', text)
    # Remove URLs
    clean_text = re.sub(r'https?://[^\s}]+', '', clean_text)
    # Remove excessive whitespace but keep single spaces
    clean_text = re.sub(r'\s+', ' ', clean_text)
    # Remove leading/trailing whitespace
    clean_text = clean_text.strip()
    return len(clean_text)

def validate_structure_basic(original: str, optimized: str) -> bool:
    """Basic LaTeX structure validation for a single point"""
    # Check \\item presence
    if '\\item' in original and '\\item' not in optimized:
        return False

    # Check major brace balance (allow 1 difference)
    orig_balance = original.count('{') - original.count('}')
    opt_balance = optimized.count('{') - optimized.count('}')

    if abs(orig_balance - opt_balance) > 1:
        return False

    return True

def validate_full_resume(original: str, optimized: str, logger=None) -> bool:
    """Validate full resume structure"""
    if logger is None:
        logger = logging.getLogger(__name__)
        
    # Check item count
    orig_items = len(re.findall(r'\\item\s', original))
    opt_items = len(re.findall(r'\\item\s', optimized))

    if orig_items != opt_items:
        logger.error(f"Item count mismatch: {orig_items} vs {opt_items}")
        return False

    # Check document structure
    for cmd in ['\\documentclass', '\\begin{document}', '\\end{document}']:
        if original.count(cmd) != optimized.count(cmd):
            logger.error(f"Document structure corrupted: {cmd}")
            return False

    logger.info("Full resume validation passed")
    return True
