from tkinter import *
from tkinter import ttk
import webbrowser
import sounddevice as sd
import soundfile as sf
import numpy as np
from PIL import Image
import librosa
import os
import shutil
import cv2
import tensorflow as tf



root = Tk()

class functions():
    def variaveis(self):
        self.name = self.name_entry.get()
        self.time = self.time_entry.get()

    def clear(self):
        self.name_entry.delete(0, END)
        self.time_entry.delete(0, END)
    
    def record(self):
        self.variaveis()
        samplerate = 48000  
        n_mels = 320
        t = self.time
        audio = sd.rec(int(samplerate * int(self.time)), samplerate=samplerate, channels=1, blocking=True)
        sd.wait()
        audio_path = "files_run/" + self.name + ".wav"
        sf.write(audio_path, audio, samplerate)

    def play(self):
        self.variaveis()
        name_file = "files_run/" + self.name + ".wav"
        data, fs = sf.read(name_file, dtype='float32')  
        sd.play(data, fs)
        status = sd.wait()

    def scale_minmax(self, X, min=0.0, max=1.0):
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max - min) + min
        return X_scaled

    def Melspec(self):
        self.variaveis()
        model = tf.keras.models.load_model('Audio_model.h5')
        classes = ['neutral','happy','sad','angry','fear']
        #classes = {'neutral':0, 'happy':1,'sad':2, 'angry':3,'fear':4}
        
        samplerate = 48000  
        n_mels = 320
        name_file = "files_run/" + self.name + ".wav"
        data, sr = librosa.load(name_file)
        mel_spec = librosa.feature.melspectrogram(y=data, sr=sr, n_mels= n_mels)
        mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40)/40
        np.save("files_run/" + self.name + '.npy', mel_db)
        smel = np.load("files_run/" + self.name + ".npy")
        img = self.scale_minmax(smel, 0, 255).astype(np.uint8)
        img = np.flip(img, axis=0) # put low frequencies at the bottom in image
        img = 255-img # invert. make black==more energy
        im = Image.fromarray(img)
        img_path = "files_run/" + self.name + ".png"
        im.save(img_path)
        image = cv2.imread(img_path)
        image = cv2.resize(image,(100,100))
        #print(image.shape)
        #image = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        image = image.reshape(-1,100,100,3)
        prediction = model.predict(image)
        #label = classes[np.argmax(prediction)]
        label = np.argmax(prediction)
        print(label)
        

    def detect(self):
        self.variaveis()
        self.Melspec()

    def quit(self): 
        #shutil.rmtree("files_run") # TO DELETE A FOLDER AND ALL FILES 
        self.root.destroy()
    
    def link1(self):
        webbrowser.open('https://github.com/edworId') # TO OPEN A LINK IN YOUR BROWSER
    
    def link2(self):
            webbrowser.open('https://github.com/edworId/speech_emotion/blob/main/README.md') # TO OPEN A LINK IN YOUR BROWSER

class Application(functions):
    def __init__(self):
        self.root = root #PRECISA DO ROOT POIS NÃO ESTÁ DENTRO DA CLASSE
        self.tela1() #PRECISA CHAMAR ANTES DO LOOP A FUNÇÃO TEAL
        self.labels()
        self.buttons()
        self.menu()
        if os.path.isdir("files_run"):
            print("Dir already exist")
        else:
            os.makedirs("files_run")
        root.mainloop() #LOOP TELA

    def tela1(self):
        self.root.title("RECORD AUDIO AND DETECT EMOTION") #TITLE
        self.root.geometry("400x300") #SETAR TAMANHO INICIAL
        self.bg = PhotoImage(file = "record.png")
        self.bground = Label(self.root, image = self.bg, bd = 0)
        self.bground.place(relx = 0, rely = 0)
        self.root.resizable(False, False) #RESPONSIVIDADE DO TAMANHO DE TELA

    def labels(self):

        ## LABEL AND ENTRY NAME
        self.lb_name = Label(self.root, text = "Name:", bg = '#6B6767', fg = '#FFFFFF', font = ('arial', 10, 'bold'))
        self.lb_name.place(relx = 0.02, rely = 0.28)

        self.name_entry = Entry(self.root)
        self.name_entry.place(relx = 0.02, rely = 0.35, relwidth = 0.3)


        # LABEL AND ENTRY TIME
        self.lb_time = Label(self.root, text = "How much time to record:", bg = '#6B6767', fg = "#FFFFFF", font = ('arial', 10, 'bold'))
        self.lb_time.place(relx = 0.5, rely = 0.28)

        self.time_entry = Entry(self.root)
        self.time_entry.place(relx = 0.5, rely = 0.35, relwidth = 0.3)

        self.lb_time = Label(self.root, text = "By Edworld", bg = '#6B6767', fg = "#FFFFFF", font = ('verdana', 8, 'bold'))
        self.lb_time.place(relx = 0.72, rely = 0.91)

    def buttons(self):
        self.bt_clear = Button(self.root, text = "Clear", bd = 4, bg = "#6B6767", fg = "#FFFFFF", font = ('arial', 10, 'bold'), command = self.clear)
        self.bt_clear.place(relx = 0.02, rely = 0.45, relwidth = 0.25, relheight = 0.1)

        self.bt_record = Button(self.root, text = "Record", bd = 4, bg = "#6B6767", fg = "#FFFFFF", font = ('arial', 10, 'bold'), command = self.record)
        self.bt_record.place(relx = 0.5, rely = 0.45, relwidth = 0.25, relheight = 0.1)

        self.bt_play = Button(self.root, text = "Play", bd = 4, bg = "#6B6767", fg = "#FFFFFF", font = ('arial', 10, 'bold'), command = self.play)
        self.bt_play.place(relx = 0.5, rely = 0.6, relwidth = 0.25, relheight = 0.1)

        self.bt_detect = Button(self.root, text = "Detect Emotion", bd = 4, bg = "#6B6767", fg = "#FFFFFF", font = ('arial', 10, 'bold'), command = self.Melspec)
        self.bt_detect.place(relx = 0.02, rely = 0.6, relwidth = 0.3, relheight = 0.1)

        self.bt_sair = Button(self.root, text = "Exit", bd = 4, bg = "#6B6767", fg = "#FFFFFF", font = ('arial', 10, 'bold'), command = self.quit)
        #self.bt_sair.place(relx = 0.7, rely = 0.8, relwidth = 0.15, relheight = 0.1)
        self.bt_sair.place(relx = 0.8, rely = 0.05, relwidth = 0.15, relheight = 0.1)

    def menu(self):
        menubar = Menu(self.root)
        self.root.config(menu=menubar)
        file1 = Menu(menubar)
        file2 = Menu(menubar)
        

        menubar.add_cascade(label = "Options", menu = file1)
        menubar.add_cascade(label = "About", menu = file2)

        file1.add_command(label = "GitHub Edworld", command = self.link1)
        file2.add_command(label = "Readme Project", command = self.link2)
    

Application()