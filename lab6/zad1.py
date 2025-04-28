from math import gcd

# Funkcja obliczająca największy wspólny dzielnik (NWD) dwóch liczb
def extended_gcd(a, b):

    if b == 0:
        return (a, 1, 0)
    else:
        gcd, x1, y1 = extended_gcd(b, a % b)
        x = y1
        y = x1 - (a // b) * y1
        return (gcd, x, y)

# Funkcja obliczająca odwrotność modulo
def mod_inverse(e, phi):

    gcd, x, _ = extended_gcd(e, phi)
    if gcd != 1:
        raise ValueError("e i phi nie są względnie pierwsze, odwrotność nie istnieje.")
    else:
        return x % phi

# Funkcja szyfrowania za pomocą klucza publicznego
def encrypt(message, public_key):
    e, n = public_key
    return pow(message, e, n)  # c = m^e mod n

# Funkcja odszyfrowania za pomocą klucza prywatnego
def decrypt(ciphertext, private_key):
    d, n = private_key
    return pow(ciphertext, d, n)  # m = c^d mod n

# Parametry wejściowe
p = 7
q = 11
e = 13
m = 2  # Wiadomość do zaszyfrowania

n = p * q
phi = (p - 1) * (q - 1)

if gcd(e, phi) != 1:
    raise ValueError("e musi być względnie pierwsze z ∅(n)")

if m >= n:
    raise ValueError("Wiadomość m musi być mniejsza od n")

d = mod_inverse(e, phi)


public_key = (e, n)
private_key = (d, n)


ciphertext = encrypt(m, public_key)


decrypted_message = decrypt(ciphertext, private_key)


print("1) n =", n, ", ∅(n) =", phi)
print("Warunek dla e:")
print("e musi być względnie pierwsze z ∅(n), czyli gcd(e, ∅(n)) = 1, oraz spełniać 1 < e < ∅(n).")
print("2) d =", d)
print("3) Klucz publiczny:", public_key)
print("   Klucz prywatny:", private_key)
print("4) Zaszyfrowana wiadomość:", ciphertext)
print("5) Odszyfrowana wiadomość:", decrypted_message)