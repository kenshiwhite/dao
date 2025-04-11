import re

def is_english(text):
    """Check if the input text contains only English letters and spaces."""
    return bool(re.match("^[A-Za-z\s]+$", text))