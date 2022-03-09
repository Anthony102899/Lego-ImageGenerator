import os
from tkinter import *
import tkinter.filedialog as tkfd
from PIL import Image
import numpy as np

def layer_interface(img_num):
    layer_names = []
    layer_nums = []
    for k in range(img_num):
        master = Toplevel()
        master.title(f"Image number {k+1}")
        master.geometry("+300+200")

        # input image and layer
        img_label = Label(master, text="Image").grid(row=0)
        layer_label = Label(master, text="Layer").grid(row=1)
        entry_img = Entry(master, width=30)
        entry_layer = Entry(master, width=30)
        entry_img.grid(row=0, column=1)
        entry_layer.grid(row=1, column=1)

        if k == img_num - 1:
            Button(master, text='Done', command=master.quit).grid(row=2, column=2, sticky=W, pady=4)
        else:
            Button(master, text='Next', command=master.quit).grid(row=2, column=2, sticky=W, pady=4)

        # Todo: Modify the default path
        # img_path = "inputs/images/"
        img_path = "new_inputs"
        img_path = os.path.join(os.path.dirname(__file__), img_path)
        path = tkfd.askopenfilename(initialdir = img_path, title = "Select file", filetypes = (("png files","*.png"),("all files","*.*")))
        entry_img.insert('0', os.path.basename(path))
        
        image = Image.open(path)
        img = PhotoImage(file=path)
        width, height = img.width(), img.height()
        if width > 250:
            scale_w = int(round(width / 250, 0))
            scale_h = int(round(height / 250, 0))
            img = img.subsample(scale_w, scale_h)
        if width < 250:
            scale_w = int(round(250 / width, 0))
            scale_h = int(round(250 / height, 0))
            img = img.zoom(scale_w, scale_h)

        Label(master, image=img).grid(row=2, column=1)
        
        mainloop()

        img_name = entry_img.get()
        img_layer = entry_layer.get()
        layer_names.append(img_name)
        layer_nums.append(img_layer)
    return layer_names, layer_nums

def show_interface():
    root = Tk()
    root.geometry("+300+300")
    
    Label(root, text="Graph", font=("", 14, "bold", "underline"), fg='#696969').grid(row=0, sticky='w')
    entry_graph = Entry(root, width=15)
    entry_graph.grid(row=0, column=1)
    graph_path = "connectivity/"
    graph_path = os.path.join(os.path.dirname(__file__), graph_path)
    path = tkfd.askopenfilename(initialdir = graph_path, title = "Select file", filetypes = (("pkl files","*.pkl"),("all files","*.*")))
    entry_graph.insert('0', os.path.basename(path))

    # input No. image and button
    Label(root, text="No. image", font=("", 14, "bold", "underline"), fg='#696969').grid(row=1, sticky='w')
    entry_num = Entry(root, width=15)
    entry_num.grid(row=1, column=1)
    Button(root, text='Next', command=root.quit).grid(row=1, column=2, sticky='e', pady=4)

    # input background color
    Label(root, text="").grid(row=2, column=1)
    Label(root, text="Background color", font=("", 14, "bold", "underline"), fg='#696969').grid(row=3, sticky='w')
    Label(root, text="R", fg='#4f4f4f').grid(row=4, column=0)
    Label(root, text="G", fg='#4f4f4f').grid(row=4, column=1)
    Label(root, text="B", fg='#4f4f4f').grid(row=4, column=2)
    entry_r = Entry(root, width=15)
    entry_g = Entry(root, width=15)
    entry_b = Entry(root, width=15)
    entry_r.grid(row=5, column=0)
    entry_g.grid(row=5, column=1)
    entry_b.grid(row=5, column=2)

    # input rotation and scaling
    Label(root, text="").grid(row=6, column=1)
    Label(root, text="Rotation degree", font=("", 14, "bold", "underline"), fg='#696969').grid(row=7, sticky='w')
    entry_degree = Entry(root, width=15, textvariable=StringVar(root, value='0'))
    entry_degree.grid(row=7, column=1)
    Label(root, text="Scale", font=("", 14, "bold", "underline"), fg='#696969').grid(row=7, column=2)
    entry_scale = Entry(root, width=15, textvariable=StringVar(root, value='1'))
    entry_scale.grid(row=7, column=3)

    # input translation
    Label(root, text="").grid(row=8, column=1)
    Label(root, text="x translation", font=("", 14, "bold", "underline"), fg='#696969').grid(row=9, sticky='w')
    entry_x = Entry(root, width=15, textvariable=StringVar(root, value='0'))
    entry_x.grid(row=9, column=1)
    Label(root, text="y translation", font=("", 14, "bold", "underline"), fg='#696969').grid(row=9, column=2)
    entry_y = Entry(root, width=15, textvariable=StringVar(root, value='0'))
    entry_y.grid(row=9, column=3)
    Label(root, text="").grid(row=9, column=1)

    mainloop()
    img_num = int(entry_num.get())
    r, g, b = entry_r.get(), entry_g.get(), entry_b.get()
    if len(r) == 0:
        r = 0
    if len(g) == 0:
        g = 0
    if len(b) == 0:
        b = 0
    if r == 0 and g == 0 and b == 0:
        rgb = []
    else:
        rgb = np.array((int(r), int(g), int(b)))
    layer_names, layer_nums = layer_interface(img_num)
    return entry_graph.get(), img_num, layer_names, layer_nums, rgb, int(entry_degree.get()), float(entry_scale.get()), int(entry_x.get()), int(entry_y.get())
    
if __name__ == '__main__':
    print(show_interface())