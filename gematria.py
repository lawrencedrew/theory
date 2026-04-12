def digital_reduction(n):
    while n > 9:
        n = sum(int(digit) for digit in str(n))
    return n

def calculate_aq(text):
    text = text.upper()
    total = 0
    for char in text:
        if 'A' <= char <= 'Z':
            total += ord(char) - ord('A') + 10
    return total

def calculate_alw(text):
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

phrase = "hyperstition"
aq_val = calculate_aq(phrase)
alw_val = calculate_alw(phrase)
ordinal_val = sum(ord(c.upper()) - 64 for c in phrase if c.isalpha())

print(f"Phrase: {phrase}")
print(f"AQ Value: {aq_val} -> Redux: {digital_reduction(aq_val)}")
print(f"ALW Value: {alw_val} -> Redux: {digital_reduction(alw_val)}")
print(f"Ordinal: {ordinal_val} -> Redux: {digital_reduction(ordinal_val)}")
