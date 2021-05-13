import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import tkinter
import PIL.Image, PIL.ImageTk
import cv2

f = open("coor.txt", "r")
coor = f.read()
canvas = None
def main():
    pass
def click(e):
    coords["x"] = e.x
    coords["y"] = e.y
    lines.append(canvas.create_line(coords["x"],coords["y"],coords["x"],coords["y"],fill='green',width=5))
    # lines.append(canvas.create_line(p1,p2,p1,p2))
def del_line(e):
    print(lines)
    canvas.delete(lines[-1])
def release(l):
    lis=[]
    recoor = []
    lis.append(coords["x"]);lis.append(coords["y"]);lis.append(coords["x2"]);lis.append(coords["x2"])
    final.append(lis)
    recoor.append(str(coords["x"]));recoor.append(str(coords["y"]));recoor.append(str(coords["x2"]));recoor.append(str(coords["y2"]))
    file = open("coor.txt", "w")
    for element in recoor:
        file.write(element+"\n")
    file.close()
    #open and read the file after the appending:
    print(coords["x"],coords["y"],coords["x2"],coords["y2"])

def drag(e):
    coords["x2"] = e.x
    coords["y2"] = e.y
    canvas.coords(lines[-1], coords["x"],coords["y"],coords["x2"],coords["y2"])
def open_file():
    global cap
    global canvas
    cap = cv2.VideoCapture("http://192.168.1.4:8081/video")
    if cap.isOpened():
        print("Device Opened\n")
    else:
        print("Failed to open Device\n")
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    width1 = int(width)
    strwidth = str(width1)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    height1 = int(height)
    strheight = str(height1)
    canvas.config(width = width, height = height) 
    window.geometry(str(strwidth)+"x"+str(strheight)) 
    
    ret,frame = get_frame(cap)
    
    if ret:
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        canvas.create_image(0, 0, image = photo, anchor = NW)
    if not pause:
        window.after(delay, play_video(cap))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_frame(cap):
      # get only one frame
    try:
        if cap.isOpened():
            print("1")
            ret, frame = cap.read()
            return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            

    except:
        messagebox.showerror(title='Video file not found', message='Please select a video file.')


def play_video():
    pause = True
    # Get a frame from the video source, and go to the next frame automatically
    ret, frame = get_frame()

    if ret:
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
        
        canvas.create_image(0, 0, image = photo, anchor = NW)

    if not pause:
        window.after(delay, play_video)

# Release the video source when the object is destroyed
def __del__():
    if cap.isOpened():
        cap.release()
window = Tk()
window.title("window_title")
window.wm_geometry("1000x500")
x = y = 0
top_frame = Frame(window)
top_frame.pack(side=TOP, pady=5)
final=[]
lines = []
bottom_frame = Frame(window)
bottom_frame.pack(side=BOTTOM, pady=5)
coords = {"x":0,"y":0,"x2":0,"y2":0}
pause = False   # Parameter that controls pause button
canvas = tk.Canvas(top_frame,cursor="cross")
canvas.pack()
p1 = p2 = 0
o = 0
btn_select=tkinter.Button(bottom_frame, text="Edit line", width=15, command=open_file)
btn_select.grid(row=0, column=0)
delay = 15   # ms
canvas.bind("<ButtonPress-1>", click)
canvas.bind("<ButtonPress-3>", del_line)
canvas.bind("<B1-Motion>", drag)
canvas.bind('<ButtonRelease-1>', release)
##### End Class #####


# Create a window and pass it to videoGUI Class
window.mainloop()