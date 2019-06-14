import random
import numpy as np

from PIL import Image, ImageColor, ImageFont, ImageDraw, ImageFilter

def generate(text, font, text_color, font_size, orientation, space_width, fit):
    if orientation == 0:
        return _generate_horizontal_text(text, font, text_color, font_size, space_width, fit)
    elif orientation == 1:
        return _generate_vertical_text(text, font, text_color, font_size, space_width, fit)
    else:
        raise ValueError("Unknown orientation " + str(orientation))

def _generate_car_plate_text_with_BG(str1,str2,str3,font1,font2,font3,type_of_plate,type_of_text):
    ##the frame of car plates is not included
    w,h = font1.getsize(str1)
    w2,h2 = font2.getsize(str2)
    w3,h3 = font3.getsize(str3)
    Len1 = len(str1)
    Len2 = len(str2)
    W= 330
    H = 90
    cont =  20 + random.randint(-10,10)
    cont2 =  30 + random.randint(-15,15)
    
    if type_of_plate == 0: #red plate
        img = Image.new("RGBA", (W, H), (197 - cont, 46 - cont, 27 - cont))
    elif type_of_plate == 1: #white plate
        img = Image.new("RGBA", (W, H), (243 - cont, 238 - cont, 248 - cont))
    elif type_of_plate == 2: #yellow plate
        img = Image.new("RGBA", (W, H), (221 - cont, 171 - cont, 22 - cont))
    else:
        raise Exception("Wrong type_of_plate")
    
    draw = ImageDraw.Draw(img)
    pos1 = (( 175-Len1*35)/2 + 140 - 15*(3 - Len2), 55-h)
    pos2 = ( 15 + (105 - 35*Len2)/2,-72)
    
    if type_of_text == 0:#black text
        draw.text(pos2, str2, font = font2, fill = (75 - cont2, 44 - cont2, 40 - cont2))
        draw.text(pos1, str1, font = font1, fill = (75 - cont2, 44 - cont2, 40 - cont2))
        draw.text(((W-w3)/2,43), str3, font = font3, fill = (75 - cont2, 44 - cont2, 40 - cont2))
    elif type_of_text == 1:#white text
        draw.text(pos2, str2, font = font2, fill = (200 - cont2, 192 - cont2, 185 - cont2))
        draw.text(pos1, str1, font = font1, fill = (200 - cont2, 192 - cont2, 185 - cont2))
        draw.text(((W-w3)/2,43), str3, font = font3, fill = (200 - cont2, 192 - cont2, 185 - cont2))
    elif type_of_text == 2: #blue text
        draw.text(pos2, str2, font = font2, fill = (66 - cont2, 98 - cont2, 133 - cont2))
        draw.text(pos1, str1, font = font1, fill = (66 - cont2, 98 - cont2, 133 - cont2))
        draw.text(((W-w3)/2,43), str3, font = font3, fill = (66 - cont2, 98 - cont2, 133 - cont2))
    elif type_of_text == 3: #green text
        draw.text(pos2, str2, font = font2, fill = (53 - cont2, 109 - cont2, 73 - cont2))
        draw.text(pos1, str1, font = font1, fill = (53 - cont2, 109 - cont2, 73 - cont2))
        draw.text(((W-w3)/2,43), str3, font = font3, fill = (53 - cont2, 109 - cont2, 73 - cont2))
    else:
        raise Exception("Wrong type_of_text")
    #del draw
    
    
    words = str2
    ch = [ch for word in words for ch in word if ch != ' ']
    sum_w = 0
    chars = []
    coords = []
    for i in range (len(ch)):
        chars.append(ch[i])
        tmp_w, tmp_h = font2.getsize(ch[i])
        x_min = pos2[0] +  sum_w
        y_min = 0#pos2[1]
        x_max = pos2[0] + sum_w + tmp_w
        y_max = tmp_h*0.5#pos2[1] + tmp_h
        coords.append((x_min, y_min, x_max, y_max))
        sum_w = sum_w + tmp_w
        #draw.rectangle((x_min,y_min,x_max,y_max), fill=None, outline=(0,0,0))

    words = str1
    ch = [ch for word in words for ch in word if ch != ' ']
    sum_w = 0
    for i in range (len(ch)):
        chars.append(ch[i])
        tmp_w, tmp_h = font1.getsize(ch[i])
        x_min = pos1[0] +  sum_w
        y_min = 0#pos2[1]
        x_max = pos1[0] + sum_w + tmp_w
        y_max = tmp_h*0.9#pos2[1] + tmp_h
        coords.append((x_min, y_min, x_max, y_max))
        sum_w = sum_w + tmp_w
        #draw.rectangle((x_min,y_min,x_max,y_max), fill=None, outline=(0,0,0))
    #img1.show()
    
    return img, coords, chars

def _generate_horizontal_text(text, font, text_color, font_size, space_width, fit):
    font = "fonts/th/sarun.ttf"
    image_font = ImageFont.truetype(font=font, size=font_size)

    words = text.split(' ')
    chars = [char for word in words for char in word]

    word_spacing = image_font.getsize(' ')[0] * space_width
    letter_spacing  = word_spacing * .3

    flatten_chars_width = [image_font.getsize(ch)[0]  for ch in chars]
    text_width =  sum(flatten_chars_width) + int(word_spacing) * (len(words) - 1)  + int(letter_spacing) * (len(chars) - (len(words)))
    text_height = max([image_font.getsize(w)[1] for w in words])

    txt_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))

    txt_draw = ImageDraw.Draw(txt_img)

    colors = [ImageColor.getrgb(c) for c in text_color.split(',')]
    c1, c2 = colors[0], colors[-1]

    fill = (
        random.randint(min(c1[0], c2[0]), max(c1[0], c2[0])),
        random.randint(min(c1[1], c2[1]), max(c1[1], c2[1])),
        random.randint(min(c1[2], c2[2]), max(c1[2], c2[2]))
    )

    coords = []
    chars = []

    for i, word in enumerate(words):  
        for j, ch in enumerate(word):
            n_char_before = sum([len(word) for word in words[:i]]) + j
            n_space_before = i
            xmin = (
                sum([char_width for char_width in flatten_chars_width[:n_char_before]]) +
                n_space_before * int(word_spacing) + 
                sum([len(word)-1 for word in words[:i]]) * letter_spacing + 
                j * letter_spacing
            )

            ymin = -(.45 * font_size)
            txt_draw.text((xmin, ymin), ch, fill=fill, font=image_font)

            #define xmax, ymax
            xmax = xmin + image_font.getsize(ch)[0]
            ymax = ymin + image_font.getsize(ch)[1]

            #reduce char height caused by font effect
            ymin = ymin + .75 * ymax

            #add margins to each side of char
            percentage_margin = 0.01
            xmin = xmin - percentage_margin * xmax
            xmax = xmax + percentage_margin * xmax
            ymin = ymin - percentage_margin * ymax
            ymax = ymax + percentage_margin * ymax

            coords.append([(xmin, ymin), (xmax, ymax)])
            chars.append(ch)
            #txt_draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)], fill = (255, 0, 0), width=2)

    if fit:
        return txt_img.crop(txt_img.getbbox()), coords
    else:
        return txt_img, coords, chars

def _generate_vertical_text(text, font, text_color, font_size, space_width, fit):
    font = "fonts/th/sarun.ttf"
    
    image_font = ImageFont.truetype(font=font, size=font_size)
    
    space_height = int(image_font.getsize(' ')[1] * space_width)

    char_heights = [image_font.getsize(c)[1] if c != ' ' else space_height for c in text]
    text_width = max([image_font.getsize(c)[0] for c in text])
    text_height = sum(char_heights)

    txt_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))

    txt_draw = ImageDraw.Draw(txt_img)

    colors = [ImageColor.getrgb(c) for c in text_color.split(',')]
    c1, c2 = colors[0], colors[-1]

    fill = (
        random.randint(c1[0], c2[0]),
        random.randint(c1[1], c2[1]),
        random.randint(c1[2], c2[2])
    )

    for i, c in enumerate(text):
        txt_draw.text((0, sum(char_heights[0:i])), c, fill=fill, font=image_font)

    if fit:
        return txt_img.crop(txt_img.getbbox())
    else:
        return txt_img
