import cv2
import numpy as np


def seperate_color(path, output):
    img = cv2.imread(path, -1)
    shape = img.shape
    colorlist = {}
    for x in range(shape[0]):
        for y in range(shape[1]):
            pixel_buffer = img[x, y]
            rgb_string = ' '.join(str(i) for i in pixel_buffer)
            if colorlist.__contains__(rgb_string):
                valuelist = colorlist.get(rgb_string)
                valuelist.append((x, y))
                colorlist[rgb_string] = valuelist
            else:
                colorlist[rgb_string] = [(x, y)]
    delete_item = []
    for key in colorlist.keys():
        # if the color is rare in image, change the color into nearest popular color (avoid too many layers)
        if len(colorlist[key]) < (shape[0] * shape[1] / 100):
            delete_item.append(key)
    for item in delete_item:
        for i in range(len(colorlist[item])):  # change rare color into nearest popular color to create smoother edge
            x_axis = colorlist[item][i][0]
            y_axis = colorlist[item][i][1]
            x_after = x_axis
            y_after = y_axis
            color_present = ' '.join(str(j) for j in img[x_after, y_after])
            while delete_item.count(color_present) != 0:
                if x_after < shape[0] - 1:
                    x_after += 1
                else:
                    y_after += 1
                color_present = ' '.join(str(j) for j in img[x_after, y_after])
            colorlist[color_present].append((x_axis, y_axis))
        del colorlist[item]
    print(colorlist.keys())
    for key in colorlist.keys():
        color = key.split(' ')
        for i in range(len(color)):
            color[i] = int(color[i])
        print(color)
        for x in range(shape[0]):
            for y in range(shape[1]):
                img[x, y] = [0, 0, 0, 0]
        for coordinate in colorlist[key]:
            img[coordinate[0], coordinate[1]] = color
        cv2.imwrite(output + path.split("/")[len(path.split("/"))-1].split(".")[0] + key + ".png", img)
    return len(colorlist.keys())


if __name__ == '__main__':
    seperate_color("images/LEGO0.png", "./output/")
