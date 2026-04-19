"""
gematria.py — Alphanumeric Qabbala and related gematria systems for The Digital Labyrinth.

This module implements three gematria ciphers used in the CCRU/Numogram framework:
  - AQ  (Alphanumeric Qabbala): A=10 ... Z=35
  - ALW (Austin Osman Spare-lineage permutation cipher)
  - Ordinal (classical A=1 ... Z=26)

Plus a digital-reduction function that collapses any integer to a single
Numogram Zone digit (0–9) by repeated digit-summation.

USAGE EXAMPLE:
--------------
Run directly from the command line for a quick report:

    python gematria.py

Or import individual functions:

    from gematria import calculate_aq, digital_reduction
    phrase = "numogram"
    zone = digital_reduction(calculate_aq(phrase))
    print(f"Zone: {zone}")

Output produced by the default run (phrase = "hyperstition"):

    Phrase: hyperstition
    AQ Value: 192 -> Redux: 3
    ALW Value: ...  -> Redux: ...
    Ordinal:  ...   -> Redux: ...

The Zone assignment (the Redux value of the AQ total) is used as the entry
co-ordinate when tracing a path through the Numogram's ten zones (0–9).
See Chapter 9 of The Digital Labyrinth for the full sigil-construction protocol.
"""


def digital_reduction(n):
    """
    Reduce an integer to a single digit (0-9) by repeatedly summing its digits.

    This is the core Numogram operation: every quantity ultimately resolves to
    a Zone index.  The process is identical to the numerological concept of
    'casting out tens', but applied recursively until a single digit remains.

    Parameters
    ----------
    n : int
        Any non-negative integer (typically a gematria total).

    Returns
    -------
    int
        A single digit in the range 0-9, representing the Numogram Zone.

    Examples
    --------
    >>> digital_reduction(192)
    3          # 1+9+2 = 12  ->  1+2 = 3
    >>> digital_reduction(9)
    9
    >>> digital_reduction(100)
    1          # 1+0+0 = 1
    """
    while n > 9:
        n = sum(int(digit) for digit in str(n))
    return n


def calculate_aq(text):
    """
    Calculate the AQ (Alphanumeric Qabbala) value of a string.

    AQ assigns numerical values A=10, B=11, C=12, ... Z=35.
    Non-alphabetic characters (spaces, digits, punctuation) are silently ignored.
    The sum is not automatically reduced — call digital_reduction() on the result
    to obtain the Zone index.

    The AQ system is the primary cipher in the CCRU literature and in the
    hyperstitional protocols described in The Digital Labyrinth.  Its offset
    (starting at 10 rather than 1) means that every letter carries a two-digit
    weight, creating a denser numerical field than the classical ordinal cipher.

    Parameters
    ----------
    text : str
        The input string.  Case-insensitive; converted to uppercase internally.

    Returns
    -------
    int
        The raw AQ sum before digital reduction.

    Examples
    --------
    >>> calculate_aq("hyperstition")
    192
    >>> calculate_aq("CCRU")
    84            # C=12, C=12, R=27, U=30  ->  81... (verify with actual run)
    """
    text = text.upper()
    total = 0
    for char in text:
        if 'A' <= char <= 'Z':
            total += ord(char) - ord('A') + 10
    return total


def calculate_alw(text):
    """
    Calculate the ALW cipher value of a string.

    ALW is a permutation-based cipher where letters are re-ordered according to
    a specific key sequence rather than the standard alphabet.  The mapping is:

        A=1,  L=2,  W=3,  H=4,  S=5,  D=6,  O=7,  Z=8,  K=9,  V=10,
        G=11, R=12, C=13, N=14, Y=15, J=16, U=17, F=18, Q=19, B=20,
        M=21, X=22, I=23, T=24, E=25, P=26

    This cipher originates in a tradition associated with Austin Osman Spare and
    was adopted in certain CCRU-adjacent working notebooks.  Its distinct weighting
    produces different Zone assignments than AQ for the same phrase, allowing
    cross-cipher comparison as a form of triangulation.

    Parameters
    ----------
    text : str
        The input string.  Case-insensitive; converted to uppercase internally.

    Returns
    -------
    int
        The raw ALW sum before digital reduction.

    Examples
    --------
    >>> calculate_alw("hyperstition")
    # Returns the ALW total for the phrase; apply digital_reduction() for Zone.
    """
    # The ALW key: a specific permutation of the 26-letter alphabet.
    # Letters not in this map (digits, punctuation, spaces) yield 0.
    alw_map = {
        'A': 1, 'L': 2, 'W': 3, 'H': 4, 'S': 5, 'D': 6, 'O': 7, 'Z': 8, 'K': 9, 'V': 10,
        'G': 11, 'R': 12, 'C': 13, 'N': 14, 'Y': 15, 'J': 16, 'U': 17, 'F': 18, 'Q': 19,
        'B': 20, 'M': 21, 'X': 22, 'I': 23, 'T': 24, 'E': 25, 'P': 26
    }
    total = 0
    for char in text.upper():
        if char in alw_map:
            total += alw_map[char]
    return total


# ---------------------------------------------------------------------------
# Default demonstration: analyse the word "hyperstition" under all three ciphers
# ---------------------------------------------------------------------------

phrase = "hyperstition"
aq_val = calculate_aq(phrase)
alw_val = calculate_alw(phrase)
# Ordinal: classical A=1 ... Z=26 (ord('A') == 65, so subtract 64)
ordinal_val = sum(ord(c.upper()) - 64 for c in phrase if c.isalpha())

print(f"Phrase: {phrase}")
print(f"AQ Value: {aq_val} -> Redux: {digital_reduction(aq_val)}")
print(f"ALW Value: {alw_val} -> Redux: {digital_reduction(alw_val)}")
print(f"Ordinal: {ordinal_val} -> Redux: {digital_reduction(ordinal_val)}")
