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
        # self.spatial_domain_example()
        # self.wait(4)
        self.spatial_domain_example_extraction()
        


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
        example_text = Text("Przykład znaku wodnego metodą LSB", font_size=36)
        example_text.to_edge(UP)

        # Load an image to demonstrate the watermarking
        image = ImageMobject("example_image.jpg")
        watermark = ImageMobject("watermark.jpg")
        watermark_bw = ImageMobject("watermark_bw.jpg")

        watermark.next_to(example_text, DOWN, buff=2)
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