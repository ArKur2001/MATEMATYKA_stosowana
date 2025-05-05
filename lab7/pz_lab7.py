def lfsr(initial_state, feedback_mask, steps=20):
    state = initial_state.copy()
    n = len(state)
    sequence = []

    for _ in range(steps):
        output = state[-1]
        sequence.append(output)
        feedback = 0
        for i in range(n):
            if feedback_mask[i]:
                feedback ^= state[i]
        state = [feedback] + state[:-1]
    return sequence

def find_period(initial_state, feedback_mask):
    state = initial_state.copy()
    n = len(state)
    seen = set()
    count = 0
    while tuple(state) not in seen:
        seen.add(tuple(state))
        feedback = 0
        for i in range(n):
            if feedback_mask[i]:
                feedback ^= state[i]
        state = [feedback] + state[:-1]
        count += 1
        if state == initial_state:
            break
    return count

def is_maximal_length(period, m):
    """
    Sprawdza, czy okres jest równy maksymalnej długości sekwencji 2^m - 1.
    """
    return period == (2**m - 1)

# Wektory sprzężenia zwrotnego:
# LFSR 1: x^4 + x + 1 -> feedback mask: [0, 0, 0, 1]
feedback1 = [1, 0, 0, 1]
# LFSR 2: x^4 + x^3 + x^2 + x + 1 -> feedback mask: [1, 1, 1, 1]
feedback2 = [1, 1, 1, 1]

# ========================
# 1) Początkowy stan: [1, 0, 0, 1]
# ========================
initial1 = [1, 0, 0, 1]
seq1a = lfsr(initial1, feedback1, 11)
seq2a = lfsr(initial1, feedback2, 11)
period1a = find_period(initial1, feedback1)
period2a = find_period(initial1, feedback2)

print("Dla wektora początkowego [1, 0, 0, 1]:")
print("LFSR 1 (x^4 + x + 1):")
print("  Sekwencja:", seq1a)
print("  Okres:", period1a)
print("  Maksymalna długość:", is_maximal_length(period1a, len(initial1)))
print("LFSR 2 (x^4 + x^3 + x^2 + x + 1):")
print("  Sekwencja:", seq2a)
print("  Okres:", period2a)
print("  Maksymalna długość:", is_maximal_length(period2a, len(initial1)))

# ========================
# 2) Początkowy stan: [1, 0, 0, 0]
# ========================
initial2 = [1, 0, 0, 0]
seq1b = lfsr(initial2, feedback1, 11)
seq2b = lfsr(initial2, feedback2, 11)
period1b = find_period(initial2, feedback1)
period2b = find_period(initial2, feedback2)

print("\nDla wektora początkowego [1, 0, 0, 0]:")
print("LFSR 1 (x^4 + x + 1):")
print("  Sekwencja:", seq1b)
print("  Okres:", period1b)
print("  Maksymalna długość:", is_maximal_length(period1b, len(initial2)))
print("LFSR 2 (x^4 + x^3 + x^2 + x + 1):")
print("  Sekwencja:", seq2b)
print("  Okres:", period2b)
print("  Maksymalna długość:", is_maximal_length(period2b, len(initial2)))