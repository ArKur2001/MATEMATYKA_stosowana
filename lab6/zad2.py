print("ZAD b)\n")

import secrets

# Krzywa eliptyczna: y^2 = x^3 + x + 6 (mod p)
p = 11

# Dane wejściowe:
a = 4  # klucz prywatny
alfa = (2, 7)  # generator
m = (3, 6)  # wiadomość

multiplication_table = {
    1: (2, 7),
    2: (5, 2),
    3: (8, 3),
    4: (10, 2),
    5: (3, 6),
    6: (7, 9),
    7: (7, 2),
    8: (3, 5),
    9: (10, 9),
    10: (8, 8),
    11: (5, 9),
    12: (2, 4),
    13: 'O'  # punkt w nieskończoności
}

def inverse_mod(k, p):
    if k == 0:
        raise ZeroDivisionError('Nie istnieje odwrotność 0 modulo p')
    return pow(k, -1, p)

def add_points(P, Q):
    if P == 'O':
        return Q
    if Q == 'O':
        return P

    x1, y1 = P
    x2, y2 = Q

    if x1 == x2 and (y1 != y2 or y1 == 0):
        return 'O'

    if P != Q:
        m = ((y2 - y1) * inverse_mod(x2 - x1, p)) % p
    else:
        m = ((3 * x1 * x1 + 1) * inverse_mod(2 * y1, p)) % p

    x3 = (m * m - x1 - x2) % p
    y3 = (m * (x1 - x3) - y1) % p

    return (x3, y3)

def mul_point(k, P):
    result = 'O'
    addend = P

    while k:
        if k & 1:
            result = add_points(result, addend)
        addend = add_points(addend, addend)
        k >>= 1

    return result

# --- Rozwiązanie ---

# 1. Klucz publiczny: alfa * a
key_public = mul_point(a, alfa)
print(f"Klucz publiczny (a * α): {key_public}")

# 2. Wybieramy losowe k
k = secrets.randbits(8)
print(f"Losowe k: {k}")

# 3. gamma = k * alfa
gamma = mul_point(k, alfa)
print(f"γ (k * α): {gamma}")

# 4. k*a*alfa
k_a = (k * a) % 13
k_a_alfa = mul_point(k_a, alfa)
print(f"k * a * α: {k_a_alfa}")

# 5. sigma = m + k*a*alfa
sigma = add_points(m, k_a_alfa)
print(f"σ (m + k*a*α): {sigma}")

# 6. Kryptogram:
print(f"Kryptogram: (γ, σ) = ({gamma}, {sigma})")

# --- Deszyfrowanie ---

# -a*gamma
minus_a_gamma = mul_point(-a % 13, gamma)  # -a mod 13
print(f"-a * γ: {minus_a_gamma}")

# -a*gamma + sigma
decrypted_m = add_points(minus_a_gamma, sigma)
print(f"Odszyfrowana wiadomość: {decrypted_m}\n")

# --- TABELKA Z ODPOWIEDZIAMI ---

print("--- WYPEŁNIONA TABELKA ---\n")
print("1. wybieramy klucz prywatny a = 4")
print("2. generatorem grupy jest α = (2,7)")
print(f"3. α ⋅a = {key_public}, zatem klucz publiczny: {key_public}")
print(f"4. chcemy zaszyfrować wiadomość m = {m}")
print(f"5. γ = {k} * {alfa} = {gamma}, σ = {m} + {k_a_alfa} = {sigma}")
print(f"6. Kryptogram: ({gamma}, {sigma})")
print(f"7. Deszyfrowanie: -a ⋅γ = {minus_a_gamma}, -a ⋅γ + σ = {decrypted_m}\n")
