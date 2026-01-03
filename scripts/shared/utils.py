import hashlib

def generate_string_id(text: str) -> str:
    """
    Generate unique ID pertaining to the input string
    
    :param text: String to give ID to
    :type text: str
    :return: Resulting ID
    :rtype: str
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()