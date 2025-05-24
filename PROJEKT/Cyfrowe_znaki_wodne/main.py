from manim import *


class Main(Scene):
    def construct(self):

        # self.title()
        self.introduction()
        self.wait(2)

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
        intro_text = Text("Czym są cyfrowe znaki wodne", font_size=48)
        intro_text.to_edge(UP)

        paragraph = Paragraph('Cyfrowy znak wodny jest to technologia służąca do oznaczania plików dźwiękowych oraz graficznych.',
                              'Metoda cyfrowego znaku wodnego polega na umieszczeniu cyfrowego sygnału znakującego',
                              'wewnątrz cyfrowej treści. Taki zapis do pliku unikalnej kombinacji bitów identyfikującej twórcę',
                              'lub właściciela majątkowych praw autorskich może stanowić trudne do wykrycia i usunięcia',
                              'zabezpieczenie. Wiąże się to jednak z pogorszeniem jakości danych zapisanych w pliku.')
        

        paragraph.width = 13
        paragraph.next_to(intro_text, DOWN, buff=2.0)
        self.play(Write(intro_text))
        self.play(FadeIn(paragraph))

