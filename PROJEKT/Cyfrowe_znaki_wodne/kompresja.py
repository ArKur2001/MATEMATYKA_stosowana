from PIL import Image

# === Ścieżki plików wejściowych ===
input_path = "example_image.jpg"          # Obraz źródłowy (pełnokolorowy)
watermark_path = "watermark.jpg"  # Obraz znaku wodnego

# === Ścieżki wyjściowe ===
output_watermarked = "watermarked_image.jpg"
output_compressed = "compressed_image.jpg"
output_extracted = "extracted_watermark.jpg"
output_extracted_compressed = "extracted_from_compressed.jpg"
output_bw_watermark = "watermark_bw.jpg"

# === Funkcja: osadzenie znaku wodnego metodą LSB ===
def embed_lsb(image, watermark):
    image = image.convert("RGB")
    watermark = watermark.convert("1").resize(image.size)
    image_pixels = image.load()
    watermark_pixels = watermark.load()
    width, height = image.size

    result = Image.new("RGB", (width, height))
    result_pixels = result.load()

    for y in range(height):
        for x in range(width):
            r, g, b = image_pixels[x, y]
            wm_bit = 1 if watermark_pixels[x, y] == 255 else 0
            b = (b & ~1) | wm_bit
            result_pixels[x, y] = (r, g, b)

    return result

# === Funkcja: wydobycie znaku wodnego z LSB ===
def extract_lsb(image):
    image = image.convert("RGB")
    width, height = image.size
    extracted = Image.new("1", (width, height))
    extracted_pixels = extracted.load()
    pixels = image.load()

    for y in range(height):
        for x in range(width):
            _, _, b = pixels[x, y]
            extracted_pixels[x, y] = 255 if (b & 1) else 0

    return extracted

# === Załaduj obrazy wejściowe ===
base_image = Image.open(input_path)
watermark_image = Image.open(watermark_path)

# === Zapisz wersję czarno-białą znaku wodnego ===
bw_watermark = watermark_image.convert("1")
bw_watermark.save(output_bw_watermark)

# === Wygeneruj obraz z LSB ===
watermarked = embed_lsb(base_image, watermark_image)
watermarked.save(output_watermarked)

# === Wygeneruj wersję skompresowaną (JPEG, niska jakość) ===
watermarked.save(output_compressed, format="JPEG", quality=20)

# === Wydobądź znak z wersji nieskompresowanej ===
extracted = extract_lsb(watermarked)
extracted.save(output_extracted)

# === Wydobądź znak z wersji skompresowanej ===
compressed = Image.open(output_compressed)
extracted_compressed = extract_lsb(compressed)
extracted_compressed.save(output_extracted_compressed)

print("✅ Wszystkie obrazy zostały wygenerowane.")
