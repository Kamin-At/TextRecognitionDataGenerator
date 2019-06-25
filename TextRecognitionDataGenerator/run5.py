import argparse
import os, errno
import random
import string
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import math
from tqdm import tqdm
from string_generator import (
    create_strings_from_dict,
    create_strings_from_file,
    create_strings_from_wikipedia,
    create_strings_randomly
)
from data_generator import CarPlateGenerator
from multiprocessing import Pool

def margins(margin):
    margins = margin.split(',')
    if len(margins) == 1:
        return [margins[0]] * 4
    return [int(m) for m in margins]

def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(description='Generate synthetic text data for text recognition.')
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="The output directory",
        default="out/",
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default=""
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), or cn (Chinese).",
        default="en"
    )
    parser.add_argument(
        "-l2",
        "--language2",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), or cn (Chinese).",
        default="en"
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created.",
        default=1000
    )
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        default=1
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Define the height of the produced images if horizontal, else the width",
        default=32,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Define the number of thread to use for image generation",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="jpg",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=int,
        nargs="?",
        help="Define skewing angle of the generated text. In positive degrees",
        default=0,
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=False,
    )
    parser.add_argument(
        "-wk",
        "--use_wikipedia",
        action="store_true",
        help="Use Wikipedia as the source text for the generation, using this paremeter ignores -r, -n, -s",
        default=False,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=int,
        nargs="?",
        help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
        default=0,
    )
    parser.add_argument(
        "-rbl",
        "--random_blur",
        action="store_true",
        help="When set, the blur radius will be randomized between 0 and -bl.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Pictures",
        default=0,
    )
    parser.add_argument(
        "-hw",
        "--handwritten",
        action="store_true",
        help="Define if the data will be \"handwritten\" by an RNN",
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
        default=0,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        help="Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        help="Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0
    )
    parser.add_argument(
        "-wd",
        "--width",
        type=int,
        nargs="?",
        help="Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used",
        default=-1
    )
    parser.add_argument(
        "-al",
        "--alignment",
        type=int,
        nargs="?",
        help="Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right",
        default=1
    )
    parser.add_argument(
        "-or",
        "--orientation",
        type=int,
        nargs="?",
        help="Define the orientation of the text. 0: Horizontal, 1: Vertical",
        default=0
    )
    parser.add_argument(
        "-tc",
        "--text_color",
        type=str,
        nargs="?",
        help="Define the text's color, should be either a single hex color or a range in the ?,? format.",
        default='#282828'
    )
    parser.add_argument(
        "-sw",
        "--space_width",
        type=float,
        nargs="?",
        help="Define the width of the spaces between words. 2.0 means twice the normal space width",
        default=1.0
    )
    parser.add_argument(
        "-m",
        "--margins",
        type=margins,
        nargs="?",
        help="Define the margins around the text when rendered. In pixels",
        default=(5, 5, 5, 5)
    )
    parser.add_argument(
        "-fi",
        "--fit",
        action="store_true",
        help="Apply a tight crop around the rendered text",
        default=False
    )


    return parser.parse_args()

def load_dict(lang):
    """
        Read the dictionnary file and returns all words in it.
    """

    lang_dict = []
    with open(os.path.join('dicts', lang + '.txt'), 'r', encoding="utf8", errors='ignore') as d:
        lang_dict = d.readlines()
    return lang_dict

def load_fonts(lang):
    """
        Load all fonts in the fonts directories
    """
    
    if lang == 'cn':
        return [os.path.join('fonts/cn', font) for font in os.listdir('fonts/cn')]
    elif lang == 'th':
        return [os.path.join('fonts/th', font) for font in os.listdir('fonts/th')]
    else:
        return [os.path.join('fonts/latin', font) for font in os.listdir('fonts/latin')]

def main():
    """
        Description: Main function
    """

    # Argument parsing
    args = parse_arguments()

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Creating word list
    lang_dict = load_dict(args.language)
    lang_dict2 = load_dict(args.language2)

    # Create font (path) list
    fonts = load_fonts(args.language)

    # Creating synthetic sentences (or word)
    os.chdir('D:\\min\\car_licence_plate\\TextRecognitionDataGenerator\\TextRecognitionDataGenerator\\dicts')
    N_per_char = math.floor(args.count/10)
    N_res = args.count - N_per_char*10
    strings = []
    with open('th4.txt','r') as f:
        for line in f:
            tmp_string = line[:-1]
            print(tmp_string)
            for i in range (N_per_char):
                strings.append(tmp_string)
        for i in range (N_res):
            strings.append(tmp_string)
            
    
    #create_strings_from_dict(args.length, args.random, args.count, lang_dict)
    strings2 = create_strings_from_dict(args.length, args.random, args.count, lang_dict2)
    
    os.chdir(args.input_file)
    
    
    font2 = ImageFont.truetype('HWYGOTH.ttf',75,encoding = "utf-8")
    font3 = ImageFont.truetype('sarun.ttf',30,encoding = "utf-8")
    #font1 = ImageFont.truetype('HWYGOTH.ttf',75,encoding = "utf-8")
    font1 = ImageFont.truetype('HWYGOTH.ttf',75,encoding = "utf-8")
    
    string_count = len(strings)
    print("string_count = " + str(string_count))
    Type_of_plates = []
    Type_of_texts = []
    Plate_numbers = []
    Direction_edge_detection = []
    DARKNESS = []
    GRADIENT = []
    gradient = []
    SHADOW = []
    BLUR = []
    BLUR_KERNEL_SIZE = []
    STD_GAUSSIAN = []
    ANGLE_TO_ROTATE = []
    for i in range (string_count):
        """tmp_rand = random.randint(0,5)
        if tmp_rand == 0:
            Type_of_plates.append(1)
            Type_of_texts.append(0)
        elif tmp_rand == 1:
            Type_of_plates.append(1)
            Type_of_texts.append(2)
        elif tmp_rand == 2:
            Type_of_plates.append(2)
            Type_of_texts.append(0)
        elif tmp_rand == 3:
            Type_of_plates.append(0)
            Type_of_texts.append(0)
        elif tmp_rand == 4:
            Type_of_plates.append(0)
            Type_of_texts.append(1)
        else:
            Type_of_plates.append(1)
            Type_of_texts.append(3)"""
            
        tmp_rand = random.randint(0,1)
        if tmp_rand == 0:
            Type_of_plates.append(1)
            Type_of_texts.append(0)
        else:
            Type_of_plates.append(2)
            Type_of_texts.append(0)
            
        tmp_rand = random.randint(0,6)
        if tmp_rand == 0:
            Plate_numbers.append(tmp_rand)
        elif tmp_rand == 1:
            Plate_numbers.append(tmp_rand)
        elif tmp_rand == 2:
            Plate_numbers.append(tmp_rand)
        elif tmp_rand == 3:
            Plate_numbers.append(tmp_rand)
        elif tmp_rand == 4:
            Plate_numbers.append(tmp_rand)
        elif tmp_rand == 5:
            Plate_numbers.append(tmp_rand)
        else:
            Plate_numbers.append(tmp_rand)

        tmp_rand = random.randint(0,3)
        if tmp_rand == 0:
            Direction_edge_detection.append(tmp_rand)
        elif tmp_rand == 1:
            Direction_edge_detection.append(tmp_rand)
        elif tmp_rand == 2:
            Direction_edge_detection.append(tmp_rand)
        else:
            Direction_edge_detection.append(tmp_rand)
            
        DARKNESS.append(-60 + random.randint(-25,25))
        GRADIENT.append(40 + random.randint(-20,20))
        STD_GAUSSIAN.append(random.randint(5 , 15))
        ANGLE_TO_ROTATE.append(random.randint(-5 , 5))
        if random.uniform(0, 1) < 0.6:
            SHADOW.append(0)
        else:
            SHADOW.append(1)
            
        if random.uniform(0, 1) < 0.7:
            gradient.append(0)
        else:
            gradient.append(1)
        
        if random.uniform(0, 1) < 0.8:
            BLUR.append(0)
            BLUR_KERNEL_SIZE.append(0)
        else:
            BLUR.append(1)
            if random.uniform(0, 1) > 0.1:
                BLUR_KERNEL_SIZE.append(3)
            else:
                BLUR_KERNEL_SIZE.append(5)
        
    for i in range (args.count):
        CarPlateGenerator.generate(
            i,
            strings[i],
            strings2[i],
            font1,
            font2,
            font3,
            args.output_dir,
            0,
            args.extension,
            Type_of_plates[i],
            Type_of_texts[i],
            Plate_numbers[i],
            Direction_edge_detection[i],
            gradient[i],
            GRADIENT[i],
            SHADOW[i],
            DARKNESS[i],
            BLUR[i],
            BLUR_KERNEL_SIZE[i],
            1,
            STD_GAUSSIAN[i],
            ANGLE_TO_ROTATE[i]
            )


    if args.name_format == 2:
        # Create file with filename-to-label connections
        with open(os.path.join(args.output_dir, "labels.txt"), 'w', encoding="utf8") as f:
            for i in range(string_count):
                file_name = str(i) + "." + args.extension
                f.write("{} {}\n".format(file_name, strings[i]))

if __name__ == '__main__':
    main()
