
import tkinter as tk
from PIL import Image,ImageDraw
import DetectNumber

WIDTH=400
HEIGHT=400

class DrawingApp:
    def __init__(self, root):
    
        self.root = root
        self.root.title("Handwriting Digit")
        self.root.geometry(f"{WIDTH}x{HEIGHT+100}")
        self.root.resizable(False,False)

        self.canvas = tk.Canvas(root, bg="black", width=WIDTH, height=HEIGHT)
        self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        # PIL create an empty image and draw object to draw on
        # memory only, not visible
        self.image = Image.new("RGB", (WIDTH, HEIGHT), "black")
        self.img_draw = ImageDraw.Draw(self.image)

    
        self.detector= DetectNumber.Detect()
    


        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.release_mouse)

        self.info=tk.Label(root,text="")
        self.info.pack(pady=5)
        clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        clear_button.pack(pady=5)


        


    def draw(self, event):
        x,y=event.x,event.y
        self.canvas.create_oval(x, y, x+20, y+20, fill="white", width=4)
        self.img_draw.ellipse([x,y,x+20,y+20],fill='white',width=4)

    
    def release_mouse(self, event):
        self.detect()


    def clear_canvas(self):
        self.info.config(text="")
        self.canvas.delete("all")
        
        self.image = Image.new("RGB", (WIDTH, HEIGHT), "black")
        self.img_draw = ImageDraw.Draw(self.image)

    def detect(self):
        predict,confidence=self.detector.predict(self.image)
        self.info.config(text=f"predict : {predict} \nconfidence : {confidence*100:.2f}%")
        # self.image.save("test.jpg")
        

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()

