from manim import *
from PIL import Image
import random

class Main(Scene):
    def construct(self):

        # self.title()
        # self.wait(2)
        # self.introduction()
        # self.wait(4)
        # self.classification()
        # self.wait(4)
        # self.spatial_domain()
        # self.wait(4)
        self.spatial_domain_example()


    def set_background(self):
        # Set a photo as the background
        bg_image =ImageMobject(r"background.jpg")  # Replace with your image path
        bg_image.scale_to_fit_height(self.camera.frame_height)
        bg_image.scale_to_fit_width(self.camera.frame_width)
        bg_image.move_to(self.camera.frame_center)
        self.add(bg_image)


    def title(self):
        self.set_background()
        title = Text("Cyfrowe Znaki Wodne", font_size=72)
        subtitle = Text("Arkadiusz Kurnik, Jan Cichoń", font_size=36)

        subtitle.next_to(title, DOWN, buff=1.0)  # Move subtitle lower under the title


        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)

    def introduction(self):
        self.set_background()
        intro_text = Text("Czym są cyfrowe znaki wodne?", font_size=48)
        intro_text.to_edge(UP)

        paragraph = Paragraph('Cyfrowy znak wodny jest to technologia służąca do oznaczania plików dźwiękowych oraz graficznych.',
                              'Metoda cyfrowego znaku wodnego polega na umieszczeniu cyfrowego sygnału znakującego',
                              'wewnątrz cyfrowej treści. Taki zapis do pliku unikalnej kombinacji bitów identyfikującej twórcę',
                              'lub właściciela majątkowych praw autorskich może stanowić trudne do wykrycia i usunięcia',
                              'zabezpieczenie. Wiąże się to jednak z pogorszeniem jakości danych zapisanych w pliku.')
        


        paragraph.width = 14
        paragraph.next_to(intro_text, DOWN, buff=2.0)
        self.play(FadeIn(intro_text))
        self.play(Write(paragraph, run_time=10))

    def classification(self):
        self.set_background()

        classification_text = Text("Klasyfikacja znaków wodnych", font_size=48)
        classification_text.to_edge(UP)
        paragraph = Paragraph(
            'Znak wodny może być klasyfikowany na podstawie różnych kryteriów,',
            'takich jak:',
            '1. Widoczność: znaki widoczne i niewidoczne',
            '2. Dziedzinę osadzenia: znaki przestrzenne, czasowe, częstotliwościowe',
            '3. Odporność: weak, robust',
            '4. Typ danych: audio, wideo, tekst, obraz'
        )


        paragraph.width = 14

        self.play(FadeIn(classification_text))
        self.play(Write(paragraph, run_time=12))

    
    def spatial_domain(self):
        self.set_background()
        spatial_text = Text("Znak wodny w dziedzinie przestrzennej", font_size=48)
        spatial_text.to_edge(UP)

        paragraph = Paragraph(
            'Znak wodny w dziedzinie przestrzennej jest umieszczany bezpośrednio w pikselach obrazu.',
            'Może być widoczny lub niewidoczny dla ludzkiego oka.',
            'najpopularniejszą metodą jest LSB (Least Significant Bit),',
            'która polega na modyfikacji najmniej znaczących bitów pikseli obrazu.',
        )

        paragraph.width = 14
        paragraph.next_to(spatial_text, DOWN, buff=2)

        self.play(FadeIn(spatial_text))
        self.play(Write(paragraph, run_time=10))

    def spatial_domain_example(self):
        self.set_background()
        example_text = Text("Przykład znaku wodnego w metodą LSB", font_size=36)
        example_text.to_edge(UP)

        # Load an image to demonstrate the watermarking
        image = ImageMobject("example_image.jpg")
        watermark = ImageMobject("watermark.jpg")


        watermark.next_to(example_text, DOWN, buff=0.5)
        watermark.scale_to_fit_height(4)
        image.scale_to_fit_height(4)
        image.to_edge(LEFT, buff=.5)
        watermark.next_to(image, RIGHT, buff=1.0)
        self.play(FadeIn(example_text))
        self.wait(2)
        self.play(FadeIn(image))
        self.wait(2)
        self.play(FadeIn(watermark))
        self.wait(4)

        c = Circle(radius=.2, color=RED)

        top_left = watermark.get_corner(UP + LEFT)
        offset = np.array([1.4, -0.5, 0])
        c.move_to(top_left + offset)

        self.play(Create(c))

        # Create a 3x3 grid of squares with random 0/255 values above the circle
        grid_size = 3
        cell_size = 0.5
        start = c.get_center() + np.array([0, 0.8, 0])  # position above the circle

        squares = VGroup()
        labels = VGroup()
        for i in range(grid_size):
            for j in range(grid_size):
                value = random.choice([0, 255])
                square = Square(side_length=cell_size)
                square.set_fill(WHITE, opacity=1)
                square.move_to(start + np.array([(j - 1) * cell_size, (1 - i) * cell_size, 0]))
                square.set_stroke(BLACK, width=1)
                label = Text(str(value), font_size=10, color=BLACK)
                label.move_to(square.get_center())
                squares.add(square)
                labels.add(label)

        self.wait(2)
        self.play(FadeIn(squares), FadeIn(labels))


if __name__ == "__main__":

    # Example: Load an image using PIL and print its size
    img = Image.open("watermark.jpg")
    print("Image size:", img.size)

    # Convert the image to black and white
    bw_img = img.convert("L")
    bw_img.save("watermarj_bw.jpg")

    # # Print RGB values of the black and white image
    # bw_pixels = bw_img.load()
    # width, height = bw_img.size
    # for y in range(height):
    #     for x in range(width):
    #         pixel_value = bw_pixels[x, y]
    #         print(f"Pixel at ({x}, {y}): {pixel_value}")

    image = Image.open("example_image.jpg")
    watermark = Image.open("watermark.jpg")

    def embed_lsb(image, watermark):
        # Convert both images to RGB and resize watermark to fit image
        image = image.convert("RGB")
        watermark = watermark.convert("1")  # Convert watermark to black and white
        watermark = watermark.resize(image.size)
        pixels = image.load()
        watermark_pixels = watermark.load()
        width, height = image.size

        result = Image.new("RGB", (width, height))
        result_pixels = result.load()

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                # Use the LSB of the blue channel to embed the watermark
                wm_bit = 1 if watermark_pixels[x, y] == 255 else 0
                b = (b & ~1) | wm_bit
                result_pixels[x, y] = (r, g, b)
        return result
    
    result = embed_lsb(image, watermark)
    result.save("watermarked_image.jpg")
    
    def extract_lsb(watermarked_image):
        watermarked_image = watermarked_image.convert("RGB")
        width, height = watermarked_image.size
        extracted = Image.new("1", (width, height))
        watermarked_pixels = watermarked_image.load()
        extracted_pixels = extracted.load()

        for y in range(height):
            for x in range(width):
                _, _, b = watermarked_pixels[x, y]
                extracted_pixels[x, y] = 255 if (b & 1) else 0
        return extracted

    extracted_watermark = extract_lsb(result)
    extracted_watermark.save("extracted_watermark.jpg")