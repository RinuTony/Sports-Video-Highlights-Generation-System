import tkinter as tk
from tkinter import filedialog
import shutil
import os

def upload_video():

    file = filedialog.askopenfilename(filetypes=[("Video Files","*.mp4")])

    if file:
        os.makedirs("videos", exist_ok=True)
        shutil.copy(file, "videos/input.mp4")
        status.config(text="Video uploaded")


def split_video():
    os.system("python split_video.py")
    status.config(text="Video split")


def extract():
    os.system("python extract_features.py")
    status.config(text="Features extracted")


def train():
    os.system("python train_model.py")
    status.config(text="Model trained")


def detect():
    os.system("python detect_highlights.py")
    status.config(text="Highlights detected")


def create():
    os.system("python create_highlight_video.py")
    status.config(text="Highlight video created")


root = tk.Tk()
root.title("Sports Highlight Generator")

tk.Button(root,text="Upload Video",command=upload_video).pack(pady=10)
tk.Button(root,text="Split Video",command=split_video).pack(pady=10)
tk.Button(root,text="Extract Features",command=extract).pack(pady=10)
tk.Button(root,text="Train Model",command=train).pack(pady=10)
tk.Button(root,text="Detect Highlights",command=detect).pack(pady=10)
tk.Button(root,text="Create Highlight Video",command=create).pack(pady=10)

status = tk.Label(root,text="Upload a video to start")
status.pack(pady=20)

root.mainloop()