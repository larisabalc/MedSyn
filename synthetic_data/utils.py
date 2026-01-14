def contains_any_keyword(text, keywords):
    """
    Return True if any keyword appears in the text.
    """
    if not text:
        return False
    text_lower = text.lower()
    return any(k in text_lower for k in keywords)
