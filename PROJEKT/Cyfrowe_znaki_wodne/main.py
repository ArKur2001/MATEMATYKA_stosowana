from manim import *
from PIL import Image
import random
import numpy as np

class Main(Scene):
    def construct(self):
        #self.title()
        #self.wait(2)
        #self.introduction()
        #self.wait(2)
        #self.classification()
        #self.wait(2)
        #self.spatial_domain()
        #self.wait(2)
        #self.spatial_domain_example()
        #self.wait(2)
        #self.lsb_weaknesses()  # Kompresja i problem na pierwszym miejscu
        #self.wait(2)
        #self.lsb_advantages()   # Zalety na osobnym slajdzie
        #self.wait(2)
        #self.lsb_disadvantages() # Wady na osobnym slajdzie
        #self.wait(2)
        self.dft_slide_overview()
        self.wait(2)
        self.dft_math_overview()
        self.wait(2)
        self.DFT_example()
        self.wait(2)
        self.DFT_compression()
        self.dft_advantages()   # Zalety na osobnym slajdzie
        self.wait(2)
        self.dft_disadvantages()

    def set_background(self):
        bg_image = ImageMobject(r"background.jpg")
        bg_image.scale_to_fit_height(self.camera.frame_height)
        bg_image.scale_to_fit_width(self.camera.frame_width)
        bg_image.move_to(self.camera.frame_center)
        self.add(bg_image)

    def title(self):
        self.set_background()
        title = Text("Cyfrowe Znaki Wodne", font_size=72)
        subtitle = Text("Arkadiusz Kurnik, Jan Cichoń", font_size=36)
        subtitle.next_to(title, DOWN, buff=1.0)
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)

    def introduction(self):
        self.set_background()
        intro_text = Text("Czym są cyfrowe znaki wodne?", font_size=48)
        intro_text.to_edge(UP)
        paragraph = Paragraph(
            'Cyfrowy znak wodny jest to technologia służąca do oznaczania plików dźwiękowych oraz graficznych.',
            'Metoda cyfrowego znaku wodnego polega na umieszczeniu cyfrowego sygnału znakującego',
            'wewnątrz cyfrowej treści. Taki zapis do pliku unikalnej kombinacji bitów identyfikującej twórcę',
            'lub właściciela majątkowych praw autorskich może stanowić trudne do wykrycia i usunięcia',
            'zabezpieczenie. Wiąże się to jednak z pogorszeniem jakości danych zapisanych w pliku.'
        )
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
            'która polega na modyfikacji najmniej znaczących bitów pikseli obrazu.'
        )
        paragraph.width = 14
        paragraph.next_to(spatial_text, DOWN, buff=2)
        self.play(FadeIn(spatial_text))
        self.play(Write(paragraph, run_time=10))

    def spatial_domain_example(self):
        self.set_background()
        example_text = Text("Przykład znaku wodnego metodą LSB", font_size=36)
        example_text.to_edge(UP)
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

        grid_size = 3
        cell_size = 0.5
        start = c.get_center() + np.array([0, 0.8, 0])
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

    def lsb_weaknesses(self):
        self.set_background()
        title = Text("Problemy podczas kompresji", font_size=48)
        title.to_edge(UP)

        wm_img = ImageMobject("watermarked_image.jpg").scale_to_fit_height(2)
        compressed_img = ImageMobject("compressed_image.jpg").scale_to_fit_height(2)
        extracted = ImageMobject("extracted_watermark.jpg").scale_to_fit_height(2)
        extracted_corrupted = ImageMobject("extracted_from_compressed.jpg").scale_to_fit_height(2)

        label1 = Text("Oryginał z LSB", font_size=24)
        label2 = Text("Po kompresji JPG", font_size=24)
        label3 = Text("Znak wodny (poprawny)", font_size=24)
        label4 = Text("Znak wodny (zniszczony)", font_size=24)

        col1 = Group(wm_img, label1, extracted, label3).arrange(DOWN, buff=0.3)
        col2 = Group(compressed_img, label2, extracted_corrupted, label4).arrange(DOWN, buff=0.3)

        all_imgs = Group(col1, col2).arrange(RIGHT, buff=1)

        all_imgs.move_to(ORIGIN)

        self.play(FadeIn(title))
        self.wait(1)
        self.play(FadeIn(all_imgs))
        self.wait(5)

    def lsb_advantages(self):
        self.set_background()
        title = Text("Zalety metody LSB", font_size=48)
        title.to_edge(UP)

        advantages = [
            "✔ Prosta implementacja",
            "✔ Niewidoczna dla oka",
            "✔ Mały wpływ na jakość obrazu",
            "✔ Niski koszt obliczeniowy"
        ]
        adv_text = VGroup(*[Text(line, font_size=30, color=GREEN) for line in advantages])
        adv_text.arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT).shift(DOWN)

        self.play(FadeIn(title))
        self.wait(1)
        self.play(LaggedStart(*[Write(t) for t in adv_text], lag_ratio=0.2))
        self.wait(5)
        self.play(FadeOut(VGroup(title, adv_text)))

    def lsb_disadvantages(self):
        self.set_background()
        title = Text("Wady metody LSB", font_size=48)
        title.to_edge(UP)

        disadvantages = [
            "✘ Łatwa do usunięcia przez kompresję (np. JPEG)",
            "✘ Nieodporna na szumy i edycję obrazu",
            "✘ Możliwa do wykrycia statystycznie",
            "✘ Niewystarczająca dla krytycznych zastosowań"
        ]
        dis_text = VGroup(*[Text(line, font_size=30, color=RED) for line in disadvantages])
        dis_text.arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT).shift(DOWN)

        self.play(FadeIn(title))
        self.wait(1)
        self.play(LaggedStart(*[Write(t) for t in dis_text], lag_ratio=0.2))
        self.wait(5)
        self.play(FadeOut(VGroup(title, dis_text)))

    def dft_slide_overview(self):
        self.set_background()
        spatial_text = Text("Cyfrowy znak wodny za pomocą DFT", font_size=48)
        spatial_text.to_edge(UP)
        paragraph = Paragraph(
            "Dyskretna Transformata Fouriera (DFT) jest matematycznym narzędziem, które pozwala przekształcić sygnał",
            "(np. obraz) z dziedziny przestrzennej (czyli wartości jasności pikseli) do dziedziny częstotliwościowej. Oznacza",
            "to, że zamiast patrzeć na obraz jako zbiór pikseli, analizujemy go jako kombinację różnych częstotliwości,",
            "które reprezentują zmienność jasności w poziomie i pionie.",
            " ",
            "W kontekście znakowania wodnego, DFT pozwala na ukrycie informacji w częstotliwościach średnich lub",
            "wysokich, które są mniej podatne na utratę w wyniku typowych operacji na obrazie, takich jak kompresja",
            "JPEG, skalowanie czy niewielkie rozmycie. Jest to ogromna zaleta w porównaniu do prostych metod jak LSB,",
            "które modyfikują piksele bezpośrednio."
        )
        paragraph.width = 14
        paragraph.next_to(spatial_text, DOWN, buff=2)
        self.play(FadeIn(spatial_text))
        self.play(Write(paragraph, run_time=10))

    def dft_math_overview(self):
        self.set_background()
        title = Text("Matematyka znaku wodnego – DFT", font_size=48)
        title.to_edge(UP)

        # Grupa 1 – Transformacja DFT
        label1 = Text("1. Transformacja Fouriera", font_size=36).set_color(YELLOW)
        eq1 = MathTex(r"F(u,v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x,y) \cdot e^{-j2\pi\left(\frac{ux}{M} + \frac{vy}{N}\right)}")
        eq2 = MathTex(r"F(u,v) = |F(u,v)| \cdot e^{j\phi(u,v)}")
        group1 = VGroup(label1, eq1, eq2).arrange(DOWN, aligned_edge=LEFT, buff=0.5).scale(0.8)

        # Grupa 2 – Wbudowanie znaku wodnego
        label2 = Text("2. Wbudowanie znaku wodnego", font_size=36).set_color(BLUE)
        eq3 = MathTex(r"|F'(u,v)| = |F(u,v)| \cdot (1 + \alpha \cdot W(u,v))")
        eq4 = MathTex(r"F'(u,v) = |F'(u,v)| \cdot e^{j\phi(u,v)}")
        group2 = VGroup(label2, eq3, eq4).arrange(DOWN, aligned_edge=LEFT, buff=0.5).scale(0.8)

        # Grupa 3 – Odwrotna transformacja i ekstrakcja
        label3 = Text("3. Odwrotna DFT i ekstrakcja znaku", font_size=36).set_color(GREEN)
        eq5 = MathTex(r"I'(x,y) = \text{IDFT}[F'(u,v)]")
        eq6 = MathTex(r"W'(u,v) = \frac{|F'_w(u,v)| - |F_o(u,v)|}{\alpha \cdot |F_o(u,v)|}")
        group3 = VGroup(label3, eq5, eq6).arrange(DOWN, aligned_edge=LEFT, buff=0.5).scale(0.8)

        # Pozycjonowanie grup
        group1.next_to(title, DOWN, buff=0.8).to_edge(LEFT)
        group2.next_to(title, DOWN, buff=0.8).to_edge(LEFT)
        group3.next_to(title, DOWN, buff=0.8).to_edge(LEFT)

        # Animacje – wyświetlenie grup po kolei
        self.play(FadeIn(title))
        self.wait(1)

        self.play(FadeIn(group1))
        self.wait(4)
        self.play(FadeOut(group1))

        self.play(FadeIn(group2))
        self.wait(4)
        self.play(FadeOut(group2))

        self.play(FadeIn(group3))
        self.wait(4)
        self.play(FadeOut(group3))

        self.play(FadeOut(title))

    def DFT_example(self):
        self.set_background()
        example_text = Text("Przykład znaku wodnego metodą DFT", font_size=36)
        example_text.to_edge(UP)
        image = ImageMobject("example_image.jpg")
        watermark = ImageMobject("watermark_bw.jpg")
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
        self.wait(2)

    def DFT_compression(self):
        self.set_background()
        title = Text("Problemy podczas kompresji", font_size=48)
        title.to_edge(UP)

        wm_img = ImageMobject("watermarked_image.jpg").scale_to_fit_height(2)
        compressed_img = ImageMobject("compressed_image.jpg").scale_to_fit_height(2)
        extracted = ImageMobject("extracted_watermark.jpg").scale_to_fit_height(2)
        extracted_corrupted = ImageMobject("watermark_bw_fixed.jpg").scale_to_fit_height(2)

        label1 = Text("Oryginał z DFT", font_size=24)
        label2 = Text("Po kompresji JPG", font_size=24)
        label3 = Text("Znak wodny (poprawny)", font_size=24)
        label4 = Text("Znak wodny (odzyskany)", font_size=24)

        col1 = Group(wm_img, label1, extracted, label3).arrange(DOWN, buff=0.3)
        col2 = Group(compressed_img, label2, extracted_corrupted, label4).arrange(DOWN, buff=0.3)

        all_imgs = Group(col1, col2).arrange(RIGHT, buff=1)

        all_imgs.move_to(ORIGIN)

        self.play(FadeIn(title))
        self.wait(1)
        self.play(FadeIn(all_imgs))
        self.wait(5)

    def dft_advantages(self):
        self.set_background()
        title = Text("Zalety metody DFT", font_size=48)
        title.to_edge(UP)

        advantages = [
            "✔ Odporność na kompresję JPEG",
            "✔ Dobra jakość po osadzeniu",
            "✔ Lepsza odporność na modyfikacje geometryczne",
            "✔ Możliwość ukrycia większej ilości danych"
        ]
        adv_text = VGroup(*[Text(line, font_size=30, color=GREEN) for line in advantages])
        adv_text.arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT).shift(DOWN)

        self.play(FadeIn(title))
        self.wait(1)
        self.play(LaggedStart(*[Write(t) for t in adv_text], lag_ratio=0.2))
        self.wait(5)
        self.play(FadeOut(VGroup(title, adv_text)))
    
    def dft_disadvantages(self):
        self.set_background()
        title = Text("Wady metody DFT", font_size=48)
        title.to_edge(UP)

        disadvantages = [
            "✘ Większa złożoność obliczeniowa",
            "✘ Trudniejsze osadzanie i ekstrakcja",
            "✘ Wrażliwość na silne zniekształcenia",
            "✘ Możliwe artefakty przy złej konfiguracji"
        ]
        dis_text = VGroup(*[Text(line, font_size=30, color=RED) for line in disadvantages])
        dis_text.arrange(DOWN, aligned_edge=LEFT).to_edge(LEFT).shift(DOWN)

        self.play(FadeIn(title))
        self.wait(1)
        self.play(LaggedStart(*[Write(t) for t in dis_text], lag_ratio=0.2))
        self.wait(5)
        self.play(FadeOut(VGroup(title, dis_text)))


# 👇 Ten kod tworzy pliki pomocnicze do animacji
if __name__ == "__main__":
    image = Image.open("example_image.jpg")
    watermark = Image.open("watermark.jpg")

    def embed_lsb(image, watermark):
        image = image.convert("RGB")
        watermark = watermark.convert("1").resize(image.size)
        pixels = image.load()
        watermark_pixels = watermark.load()
        width, height = image.size

        result = Image.new("RGB", (width, height))
        result_pixels = result.load()

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                wm_bit = 1 if watermark_pixels[x, y] == 255 else 0
                b = (b & ~1) | wm_bit
                result_pixels[x, y] = (r, g, b)
        return result

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

    watermarked = embed_lsb(image, watermark)
    watermarked.save("watermarked_image.jpg")

    # Zapis skompresowanego obrazu (JPEG, niska jakość)
    watermarked.save("compressed_image.jpg", format="JPEG", quality=20)

    # Wydobycie znaku wodnego z oryginalnego
    extract_lsb(watermarked).save("extracted_watermark.jpg")

    # Wydobycie z wersji po kompresji
    compressed = Image.open("compressed_image.jpg")
    extract_lsb(compressed).save("extracted_from_compressed.jpg")
