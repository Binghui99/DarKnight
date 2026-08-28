"""
utils/solve_math.py
Detects and solves simple arithmetic equations extracted from CAPTCHA OCR text.
"""
import re

# Error-token sentinels returned by solve_equation() on failure
ERROR_EMPTY     = "ERROR_EMPTY"
DIV_ZERO        = "DIV_ZERO"
MATH_ERR_PREFIX = "MATH_ERR"


def is_math_equation(text: str) -> bool:
    """Return True if the OCR string looks like a mathematical expression."""
    return bool(re.search(r"[\+\-\*\/=×÷]|\?|\d[xX]\d", text))


def is_valid_math_ocr(text: str) -> bool:
    """
    Stricter check used after OCR to decide whether to attempt solving.
    Both conditions must hold:
      1. Contains = or ?  (marks the unknown or answer position)
      2. Contains at least one arithmetic operator: + - * / × ÷
    """
    has_eq_or_q  = "=" in text or "?" in text
    has_operator = bool(re.search(r"[+\-*/×÷]", text))
    return has_eq_or_q and has_operator


def solve_equation(text: str) -> str:
    """
    Parse and evaluate a simple arithmetic expression from raw OCR text.

    Handles:
    - Answer already embedded: "17+16=33" → returns "33" directly
    - Unknown marker:          "4+5=?"    → strips marker, evaluates
    - Trailing equals:         "34-66="   → strips "=", evaluates
    - Unicode superscript digits
    - Common OCR mis-reads (o→0, l→1, …)
    - Division-by-zero

    Returns the string result, or an error token on failure.
    """
    if not text:
        return ERROR_EMPTY

    # If the answer is already embedded after "=" (e.g. "17+16=33"), extract it.
    eq_answer = re.search(r"=\s*(\d+)\s*$", text.strip())
    if eq_answer:
        return eq_answer.group(1)

    if not is_math_equation(text):
        return text

    # 1. Clean and normalise
    clean = text.replace(" ", "").replace("?", "").replace("=", "")

    # Translate unicode superscript digits to ASCII
    unicode_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
    clean = clean.translate(unicode_map)

    print(f"[solve_math] cleaned input: {clean}")

    # Fix common OCR character confusion
    replacements = {
        "o": "0", "l": "1", "b": "6", "z": "2",
        "s": "5", "g": "9",
        "×": "*", "x": "*", "÷": "/", "：": ":",
        "×*": "*", "x*": "*", "**": "*",
        "^": "",   # no exponentiation in CAPTCHA math
    }
    clean = clean.lower()
    for old, new in replacements.items():
        clean = clean.replace(old, new)

    print(f"[solve_math] after replacements: {clean}")

    # 2. Handle chained expression (e.g. "3+4+5", "12*3-2", "10/2")
    chain_match = re.search(r"\d+(?:[\+\-\*\/]\d+)+", clean)
    if chain_match:
        expr = chain_match.group()
        # only digits and basic operators allowed
        if re.fullmatch(r"[\d\+\-\*\/]+", expr):
            # Use integer division (//) 
            int_expr = expr.replace("/", "//")
            try:
                result = eval(int_expr)  # noqa: S307
                return str(result)
            except ZeroDivisionError:
                return DIV_ZERO
            except Exception as exc:
                return f"{MATH_ERR_PREFIX}_{str(exc)[:5]}"

    # 3. Fall back: return the last number found in cleaned text, or cleaned text
    numbers = re.findall(r"\d+", clean)
    if numbers:
        return numbers[-1]

    return clean.strip()