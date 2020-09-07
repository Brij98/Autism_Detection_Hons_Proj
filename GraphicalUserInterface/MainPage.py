import tkinter as tk
from tkinter import ttk
from tkinter.ttk import Frame

root = tk.Tk()
root.title("Autism Detection using Eye Tracking Data")

tab_Control = ttk.Notebook(root)

tab1 = ttk.Frame(tab_Control)
tab2 = ttk.Frame(tab_Control)

tab_Control.add(tab1, text="Upload Files")
tab_Control.add(tab2, text="Report")

tab_Control.pack(expand=1, fill="both")

ttk.Label(tab1, text="Upload Files", size='50').grid(column=0, row=0, padx=10, pady=10)

ttk.Label(tab2, text="Prediction Results").grid(column=0, row=0, padx=10, pady=10)

root.mainloop()


class MainPage(Frame):
    def __init__(self):
        super().__init__()
        self.initUI()


def main():
    root = tk()
    root.geometry('1024x600')
    app = MainPage()
    root.mainloop()


if __name__ == '__main__':
    main()