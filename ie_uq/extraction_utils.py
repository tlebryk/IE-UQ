import re


# TODO: move me to shared library code
def get_text_between_curly_braces(input_string):
    match = re.search(r"\{.*\}", input_string, re.DOTALL)
    return match.group(0) if match else None
