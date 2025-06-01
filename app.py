import tkinter as tk
from threading import Event, Lock, Thread
from tkinter import ttk

import numpy as np
import pyaudio
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from models import *

RESPONSE_TIME = 1
RATE = 16000
CHANNELS = 1
CHUNK = int(RATE * RESPONSE_TIME)
FORMAT = pyaudio.paFloat32


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Drone Detector")
        self.configure(background="white")

        label = ttk.Label(self, background="white")
        indicators = ttk.Frame(self)

        models: list[type[AudioClassificationModel]] = [M5, VGG, CRNN, Wav2Vec]
        name_to_model = {m.__name__: m for m in models}
        models_names = list(name_to_model.keys())
        models_menu = ttk.OptionMenu(
            self, tk.StringVar(self), models_names[0], *models_names, command=self.set_model
        )

        fig = Figure((3, 2))
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas_widget = canvas.get_tk_widget()

        models_menu.pack()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        label.pack()
        indicators.pack(fill=tk.BOTH)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: AudioClassificationModel | None = None
        self.name_to_model = name_to_model
        self.label = label
        self.canvas = canvas
        self.indicators = indicators
        self.thread = Thread(target=self.process)
        self.model_lock = Lock()
        self.stop_event = Event()

        self.set_model(models_names[0])

    def mainloop(self, n=0):
        self.thread.start()
        super().mainloop(n)

    def destroy(self):
        self.stop_event.set()
        self.wait_thread_and_destroy()

    def wait_thread_and_destroy(self):
        if self.thread.is_alive():
            self.after(100, self.wait_thread_and_destroy)
        else:
            super().destroy()

    def set_model(self, model_name: str):
        with self.model_lock:
            checkpoint_path = f"checkpoints/{model_name}.ckpt"
            model_class = self.name_to_model[model_name]
            self.model = model_class.load_from_checkpoint(checkpoint_path).to(self.device).eval()

    def process(self):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=0,
        )

        try:
            with torch.no_grad():
                while stream.is_active() and not self.stop_event.is_set():
                    frames = stream.read(CHUNK)

                    buffer = np.frombuffer(frames, dtype=np.float32)
                    data = torch.from_numpy(buffer)
                    input_data = data.unsqueeze(0).unsqueeze(0).to(self.device)

                    with self.model_lock:
                        logits = self.model(input_data)
                        result = self.model.decode(logits)[0]

                    self.after(0, self.update_frame, data, result)
        finally:
            stream.close()
            audio.terminate()

    def update_frame(self, data, result):
        for child in self.indicators.winfo_children():
            child.destroy()

        for i, (name, probability) in enumerate(result.items()):
            prob = round(probability,2 )
            text = f"{name}\n{prob}"
            color = "red" if prob < 0.5 else "green"
            label = ttk.Label(self.indicators, text = text, background=color, borderwidth=1, relief="solid")
            label.pack(fill=tk.BOTH)

        ax = self.canvas.figure.gca()
        ax.clear()
        ax.plot(data, color="black")
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)

        self.canvas.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()
