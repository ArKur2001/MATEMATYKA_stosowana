import numpy as np
from PIL import Image
import cv2

input_path = "example_image.jpg"
watermark_path = "watermark.jpg"

output_watermarked = "watermarked_dft_fixed.jpg"
output_compressed = "compressed_dft_fixed.jpg"
output_extracted = "extracted_watermark_dft_fixed.jpg"
output_extracted_compressed = "extracted_from_compressed_dft_fixed.jpg"
output_bw_watermark = "watermark_bw_fixed.jpg"

alpha = 5  # zmniejszone, można zwiększać eksperymentalnie

def create_circular_mask(shape, inner_radius, outer_radius):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - center_col)**2 + (Y - center_row)**2)
    mask = (dist_from_center >= inner_radius) & (dist_from_center <= outer_radius)
    return mask

def embed_dft_fixed(image, watermark):
    image = image.convert("L")
    watermark = watermark.convert("1").resize(image.size)

    img_array = np.array(image).astype(np.float32)
    wm_array = np.array(watermark) // 255  # 0 lub 1

    dft = np.fft.fft2(img_array)
    dft_shift = np.fft.fftshift(dft)

    magnitude = np.abs(dft_shift)
    phase = np.angle(dft_shift)

    mask = create_circular_mask(img_array.shape, inner_radius=30, outer_radius=60)

    # Przygotuj maskę do watermarka (dopasuj rozmiar)
    wm_resized = wm_array.astype(np.float32)

    # Na miejscach maski, pomnóż amplitudę przez (1 + alpha * watermark bit)
    new_magnitude = magnitude.copy()
    # Ustawiamy watermark tylko tam gdzie maska == True
    new_magnitude[mask] = magnitude[mask] * (1 + alpha * wm_resized[mask])

    # Odwrotna transformata
    new_dft = new_magnitude * np.exp(1j * phase)
    idft = np.fft.ifft2(np.fft.ifftshift(new_dft))
    img_watermarked = np.real(idft)

    # Klipowanie do 0-255 i konwersja do uint8
    img_watermarked = np.clip(img_watermarked, 0, 255).astype(np.uint8)
    return Image.fromarray(img_watermarked)

def extract_dft_fixed(original, watermarked):
    original = original.convert("L")
    watermarked = watermarked.convert("L")

    orig_array = np.array(original).astype(np.float32)
    watermarked_array = np.array(watermarked).astype(np.float32)

    dft_orig = np.fft.fftshift(np.fft.fft2(orig_array))
    dft_watermarked = np.fft.fftshift(np.fft.fft2(watermarked_array))

    mag_orig = np.abs(dft_orig)
    mag_watermarked = np.abs(dft_watermarked)

    mask = create_circular_mask(orig_array.shape, inner_radius=30, outer_radius=60)

    diff = np.zeros_like(mag_orig, dtype=np.float32)
    diff[mask] = (mag_watermarked[mask] / (mag_orig[mask] + 1e-10)) - 1
    diff[mask] = diff[mask] / alpha

    # Normalizacja do 0-255 dla wyświetlenia
    diff_norm = np.zeros_like(diff)
    diff_norm[mask] = diff[mask]

    # Skalowanie do [0,255]
    diff_norm = (diff_norm - diff_norm.min()) / (diff_norm.max() - diff_norm.min() + 1e-10)
    diff_norm = (diff_norm * 255).astype(np.uint8)

    # Progowanie i filtracja medianowa, aby usunąć szum
    _, thresh = cv2.threshold(diff_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    filtered = cv2.medianBlur(thresh, 3)

    return Image.fromarray(filtered)

# === Główna część ===
base_image = Image.open(input_path)
watermark_image = Image.open(watermark_path)

bw_watermark = watermark_image.convert("1")
bw_watermark.save(output_bw_watermark)

watermarked = embed_dft_fixed(base_image, watermark_image)
watermarked.save(output_watermarked)

watermarked.save(output_compressed, format="JPEG", quality=20)

extracted = extract_dft_fixed(base_image, watermarked)
extracted.save(output_extracted)

compressed = Image.open(output_compressed)
extracted_compressed = extract_dft_fixed(base_image, compressed)
extracted_compressed.save(output_extracted_compressed)

print("✅ Poprawione DFT watermarking zadziałało.")
