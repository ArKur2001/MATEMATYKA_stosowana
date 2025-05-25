from manim import *
from PIL import Image
import random
import numpy as np

class Main(Scene):
    def construct(self):
        self.title()
        self.wait(2)
        self.introduction()
        self.wait(10)
        self.classification()
        self.wait(10)
        self.spatial_domain()
        self.wait(10)
        self.spatial_domain_example()
        self.wait(10)
        # self.spatial_domain_example_extraction()
        # self.wait(2)
        # self.lsb_weaknesses()  # Kompresja i problem na pierwszym miejscu
        # self.wait(2)
        # self.lsb_advantages()   # Zalety na osobnym slajdzie
        # self.wait(2)
        # self.lsb_disadvantages() # Wady na osobnym slajdzie
        # self.wait(2)
        # self.dft_math_overview()
        # self.wait(2)
        # self.DFT_example()
        # self.wait(2)
        # self.DFT_compression()
        # self.dft_advantages()   # Zalety na osobnym slajdzie
        # self.wait(2)
        # self.dft_disadvantages()

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
        self.play(Write(paragraph, run_time=12))

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
            'najprostrzą metodą jest LSB (Least Significant Bit),',
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
        watermark_bw = ImageMobject("watermark_bw.jpg")

        watermark.next_to(example_text, DOWN, buff=2)
        watermark.next_to(example_text, DOWN, buff=0.5)
        watermark.scale_to_fit_height(4)
        image.scale_to_fit_height(4)
        image.to_edge(LEFT, buff=.5)
        image.to_edge(DOWN, buff=.5)
        watermark.next_to(image, RIGHT, buff=1.0)
        self.play(FadeIn(example_text))
        self.wait(2)
        self.play(FadeIn(image))
        self.wait(2)
        self.play(FadeIn(watermark))
        self.wait(2)
        c = Circle(radius=.2, color=RED)
        top_left = watermark.get_corner(UP + LEFT)
        offset = np.array([1.4, -0.5, 0])
        c.move_to(top_left + offset)
        self.play(Create(c))

        # Create a 3x3 grid of squares with random 0/255 values above the circle
        cell_size = 1
        start = c.get_center() + np.array([-0.5, 1.5, 0])  # position above the circle

        grid1 = [
            [[255,255,255],[255,255,255],[255,255,255]],
            [[255,255,255],[64,255,64],[64,255,64]],
            [[255,255,255],[64,255,64],[64,255,64]],
            ]
        
        grid2 = [
            [[255,255,255],[255,255,255],[255,255,255]],
            [[255,255,255],[0,0,0],[0,0,0]],
            [[255,255,255],[0,0,0],[0,0,0]],
            ]

        squares1, labels1 = self.create_grid(grid1, cell_size, start)
        squares2, labels2 = self.create_grid(grid2, cell_size, start)
        squares2.move_to(squares1.get_center())
        labels2.move_to(labels1.get_center())



        self.wait(2)
        self.play(FadeIn(squares1), FadeIn(labels1))

        self.wait(2)

        watermark_bw.scale_to_fit_height(4)
        watermark_bw.move_to(watermark.get_center())
        self.play(Transform(watermark, watermark_bw), Transform(squares1, squares2), Transform(labels1, labels2))


        self.wait(4)

        # Show the code for extract_lsb as a text paragraph
        code = '''
    def embed_lsb(image, watermark):
        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                # Use the LSB of the blue channel to embed the watermark
                wm_bit = 1 if watermark_pixels[x, y] == 255 else 0
                b = (b & ~1) | wm_bit
                result_pixels[x, y] = (r, g, b)
    '''
        rendered_code = Code(
            code_string=code,
            language="python",
            background="window",
            formatter_style="native")
        
        print(rendered_code.get_styles_list())

        rendered_code.scale(0.5)
        rendered_code.to_edge(DOWN, buff=1.0)

        self.play(FadeIn(rendered_code))

        self.wait(4)

        c2 = Circle(radius=.2, color=RED)

        top_left = image.get_corner(UP + LEFT)
        offset = np.array([1.4, -0.5, 0])
        c2.move_to(top_left + offset)

        self.play(Create(c2))

        start2 = c2.get_center() + np.array([-0.5, 1.5, 0])  # position above the circle

        
        grid3 = [
            [[54,55,35],[53,55,31],[52,56,30]],
            [[55,51,33],[52,57,36],[57,51,37]],
            [[50,52,32],[51,59,32],[59,52,34]],
            ]


        
        squares3, labels3 = self.create_grid(grid3, cell_size, start2)

        self.play(FadeIn(squares3), FadeIn(labels3))



        for i, square in enumerate(squares2):
            square1_center = square.get_center()
            square3_center = squares3[i].get_center()
            arrow = Arrow(start=square1_center, end=square3_center, buff=0.1, color=YELLOW)

            self.play(Create(arrow))
            self.remove(arrow)
            number_to_modify = labels3[(i*3)+2]
            number_from_old_grid = labels1[(i*3)+2]


            if number_from_old_grid.text == "255":
                new_text = str(int(number_to_modify.text) + 1)
            else:
                new_text = number_to_modify.text

            number_to_modify.become(Text(new_text, font_size=15, color=number_to_modify.color).move_to(number_to_modify.get_center()))

        self.wait(2)


        watermarked_image = ImageMobject("watermarked_image.jpg")
        
        watermarked_image.scale_to_fit_height(4)
        watermarked_image.next_to(watermark, RIGHT, buff=1.0)

        self.play(FadeIn(watermarked_image))



    def spatial_domain_example_extraction(self):
        self.set_background()
        example_text = Text("Przykład ekstrakcji znaku wodnego metodą LSB", font_size=36)
        example_text.to_edge(UP)

        # Load the watermarked image
        watermarked_image = ImageMobject("watermarked_image.jpg")
        extracted_watermark = ImageMobject("extracted_watermark.jpg")

        self.add(example_text)
        watermarked_image.scale_to_fit_height(4)
        watermarked_image.to_edge(LEFT, buff=.5)
        watermarked_image.to_edge(DOWN, buff=.5)

        extracted_watermark.scale_to_fit_height(4)
        extracted_watermark.to_edge(RIGHT, buff=.5)
        extracted_watermark.to_edge(DOWN, buff=.5)

        self.add(watermarked_image)

        code = '''
    def extract_lsb(watermarked_image):
        for y in range(height):
            for x in range(width):
                _, _, b = watermarked_pixels[x, y]
                extracted_pixels[x, y] = 255 if (b & 1) else 0
        return extracted
    '''
        rendered_code = Code(
            code_string=code,
            language="python",
            background="window",
            formatter_style="native")
    


        rendered_code.scale(0.5)
        rendered_code.to_edge(DOWN, buff=1.0)

        self.play(FadeIn(rendered_code))


        c2 = Circle(radius=.2, color=RED)

        top_left = watermarked_image.get_corner(UP + LEFT)
        offset = np.array([1.4, -0.5, 0])
        c2.move_to(top_left + offset)

        self.play(Create(c2))

        start2 = c2.get_center() + np.array([-0.5, 1.5, 0])  # position above the circle

        
        grid3 = [
            [[54,55,36],[53,55,32],[52,56,31]],
            [[55,51,34],[52,57,36],[57,51,37]],
            [[50,52,33],[51,59,32],[59,52,34]],
            ]

        cell_size = 1

        
        squares3, labels3 = self.create_grid(grid3, cell_size, start2)
        squares5, labels5 = self.create_grid(grid3, cell_size, start2)

        self.play(FadeIn(squares3), FadeIn(labels3))
        self.add(squares5, labels5)



        self.wait(5)

        squares4, labels4 = self.create_grid(grid3, cell_size, start2)

        squares4.move_to(squares3.get_center() + np.array([7, 0, 0]))
        labels4.move_to(squares4.get_center())

        self.play(Transform(squares3, squares4), Transform(labels3, labels4))

        grid2 = [
            [[0,0,1],[0,0,1],[0,0,1]],
            [[0,0,1],[0,0,0],[0,0,0]],
            [[0,0,1],[0,0,0],[0,0,0]],
            ]

        squares6, labels6 = self.create_grid(grid2, cell_size, start2)
        squares6.move_to(squares3.get_center())
        labels6.move_to(squares4.get_center())
        self.play(Transform(squares4, squares6), Transform(labels4, labels6))

        grid2 = [
            [[255,255,255],[255,255,255],[255,255,255]],
            [[255,255,255],[0,0,0],[0,0,0]],
            [[255,255,255],[0,0,0],[0,0,0]],
            ]

        squares7, labels7 = self.create_grid(grid2, cell_size, start2)
        squares7.move_to(squares3.get_center())
        labels7.move_to(squares4.get_center())
        self.play(Transform(squares6, squares7), Transform(labels6, labels7))



        self.play(FadeIn(extracted_watermark))


    def create_grid(self, grid,cell_size, start):

        squares = VGroup()
        labels = VGroup()


        for i, row in enumerate(grid):
            for j, collumn in enumerate(row):
                square = Square(side_length=cell_size)
                square.set_fill(color.rgb_to_color(collumn), opacity=1)
                square.move_to(start + np.array([(j - 1) * cell_size, (1 - i) * cell_size, 0]))
                square.set_stroke(BLACK, width=1)

                    
                label0 = Text(str(collumn[0]), font_size=15, color=BLACK )
                label1 = Text(str(collumn[1]), font_size=15, color=BLACK )
                label2 = Text(str(collumn[2]), font_size=15, color=BLACK )


                if collumn[0] < 127  and collumn[1] < 127  and collumn[2] < 127 :
                    label0.set_color(WHITE)
                    label1.set_color(WHITE)
                    label2.set_color(WHITE)

                label0.move_to(square.get_center())
                label1.next_to(label0, DOWN, buff=0.1)
                label2.next_to(label0, UP, buff=0.1)
                squares.add(square)
                labels.add(label0)
                labels.add(label1)
                labels.add(label2)

        return squares, labels
    
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

    # Example: Load an image using PIL and print its size
    img = Image.open("watermark.jpg")
    print("Image size:", img.size)

    # Convert the image to black and white
    bw_img = img.convert("L")
    bw_img.save("watermark_bw.jpg")

    # # Print RGB values of the black and white image
    # bw_pixels = img.load()
    # width, height = bw_img.size
    # for y in range(height):
    #     for x in range(width):
    #         pixel_value = bw_pixels[x, y]
    #         print(f"Pixel at ({x}, {y}): {pixel_value}")

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
    


    def embed_lsb_variable(image, watermark, depth):
        image = image.convert("RGB")
        watermark = watermark.convert("1").resize(image.size)
        pixels = image.load()
        watermark_pixels = watermark.load()
        width, height = image.size

        result = Image.new("RGB", (width, height))
        result_pixels = result.load()


        bitmask = ((2**8)-1) - ((2**depth) - 1)
        print("Bitmask:", bin(bitmask))

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                wm_bit = 1 if watermark_pixels[x, y] == 255 else 0
                r = (r & bitmask) | wm_bit
                g = (g & bitmask) | wm_bit
                b = (b & bitmask) | wm_bit
                    # print(bin(~((2**depth))))


                # print(wm_bit*(2**depth-1))
                result_pixels[x, y] = (r, g, b)
        return result

    def extract_lsb_variable(watermarked_image, depth):
        watermarked_image = watermarked_image.convert("RGB")
        width, height = watermarked_image.size
        extracted = Image.new("1", (width, height))
        watermarked_pixels = watermarked_image.load()
        extracted_pixels = extracted.load()

        for y in range(height):
            for x in range(width):
                r, g, b = watermarked_pixels[x, y]
                if (r & (2**depth-1)) == 0 or (g & (2**depth-1)) == 0 or (b & (2**depth-1)) == 0:
                    extracted_pixels[x, y] = 255
                else:
                    extracted_pixels[x, y] = 0

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


    for n in range(1, 8):
        watermarked = embed_lsb_variable(image, watermark, n)
        watermarked.save(f"watermarked_image{n}.jpg")
        extract_lsb_variable(watermarked, n).save(f"extracted_watermark{n}.jpg")

