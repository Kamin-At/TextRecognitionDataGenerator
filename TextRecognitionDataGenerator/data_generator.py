import os
import random
import math

from imgaug import augmenters as aug
from PIL import Image, ImageFilter, ImageDraw

import computer_text_generator
import background_generator
import distorsion_generator
import xml_util
import numpy as np

try:
    import handwritten_text_generator
except ImportError as e:
    print('Missing modules for handwritten text generation.')

class CarPlateGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(cls, index, text, text2, font1, font2, font3, out_dir, name_format, extension, type_of_plate, type_of_text, frame_number, direction_edge_detection, gradient, GRADIENT, Shadow, DARKNESS, blur, blur_kernel_size, gaussian_Noise, std_gaussian, angle_to_rotate):
        image = None
        strings = text.split()
        print(strings)
        str1 = strings[1]
        str2 = strings[0]
        
        ##########################
        # Create picture of text #
        ##########################
        
        txt_img, coords, chars = computer_text_generator._generate_car_plate_text_with_BG(str1,str2,text2,font1,font2,font3,type_of_plate,type_of_text)
        
        
        
        #############################
        # Generate background image #
        #############################
        
        BG = background_generator.car_plate_frame(frame_number)
        
        #############################
        # Place text with alignment #
        #############################

        W= 330
        H = 90
        
        kernel_size = 3
        ###apply random direction of edge detection
        if direction_edge_detection == 0:
            kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        elif direction_edge_detection == 1:
            kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        elif direction_edge_detection == 2:
            kernel = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        else:
            kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

        kernel = ImageFilter.Kernel(kernel.shape, kernel = kernel.flatten(), scale=kernel_size**2 , offset=0)
        img22 = txt_img.filter(kernel)
        img22 = img22.convert(mode = 'L')
        #img22.show()
        Arr = np.array(txt_img)
        Arr2 = np.array(img22)

        for i in range(Arr2.shape[0]):
            for j in range(Arr2.shape[1]):
                if Arr2[i][j] > 10:
                    Arr[i,j,:] = np.clip( Arr[i,j,:] + 120, a_min = 0, a_max = 255)
        img = Image.fromarray(Arr)
        
        draw = ImageDraw.Draw(img)
        Arr = np.array(img)
        
        ################
        ### gradient ###
        ################
        
        if gradient:
            i_m = random.randint(0,Arr[:,:,0].shape[0])
            j_m = random.randint(0,Arr[:,:,0].shape[1])
            
            if i_m == j_m:
                if random.randint(0,1):
                    i_m = i_m + 1
                else:
                    j_m = j_m + 1
            c = random.randint(0,1)
            x = np.zeros((Arr[:,:,0].shape[0],Arr[:,:,0].shape[1]))
            a = 1/(i_m - j_m)
            b = a
            k = i_m/(i_m - j_m)
            if c:
                for i in range (Arr[:,:,0].shape[0]):
                    for j in range (Arr[:,:,0].shape[1]):
                        x[i][j] = a*(i-i_m) + b*(j-j_m) + k
            else:
                for i in range (Arr[:,:,0].shape[0]):
                    for j in range (Arr[:,:,0].shape[1]):
                        x[i][j] = -a*(i-i_m) - b*(j-j_m) + k
            Max = np.amax(x)
            Min = np.amin(x)
            x = (x-np.full((Arr[:,:,0].shape[0],Arr[:,:,0].shape[1]),Min))/(Max - Min)
            x = x*GRADIENT

            Arr[:,:,0] = np.clip(Arr[:,:,0] + x, a_min = 0, a_max = 255) 
            Arr[:,:,1] = np.clip(Arr[:,:,1] + x, a_min = 0, a_max = 255) 
            Arr[:,:,2] = np.clip(Arr[:,:,2] + x, a_min = 0, a_max = 255) 
        #####################################
        # Combine background and text_image #
        #####################################   
        img = Image.fromarray(Arr)
        
        if frame_number == 0:
            new_w = 136
            new_h = 54#for 4.jpg
        elif frame_number == 1:
            new_w = 165
            new_h = 60#for 3.jpg
        elif frame_number == 2:
            new_w = 175
            new_h = 67#for 10.jpg
        elif frame_number == 3:
            new_w = 117
            new_h = 55#for 19.jpg
        elif frame_number == 4:
            new_w = 387
            new_h = 165#for 18_1.jpg
        elif frame_number == 5:
            new_w = 120
            new_h = 51#for 20.jpg
        elif frame_number == 6:
            new_w = 590
            new_h = 190#for 21.jpg
            
        else:
            raise Exception("Wrong Frame_number")
            
        img = img.resize( (new_w,new_h), resample=0)
        
        if frame_number == 0:
            BG.paste(img, box=(34,20), mask=None)#for 4.jpg
        elif frame_number == 1:
            BG.paste(img, box=(15,17), mask=None)#for 3.jpg
        elif frame_number == 2:
            BG.paste(img, box=(13,9), mask=None)#for 10.jpg
        elif frame_number == 3:
            BG.paste(img, box=(92,35), mask=None)#for 19.jpg
        elif frame_number == 4:
            BG.paste(img, box=(4,4), mask=None)#for 18_1.jpg
        elif frame_number == 5:
            BG.paste(img, box=(70,57), mask=None)#for 20.jpg
        elif frame_number == 6:
            BG.paste(img, box=(5,5), mask=None)#for 21.jpg
        else:
            raise Exception("Wrong Frame_number")
        
        coords = coordinate_resize(W, H, new_w, new_h, frame_number ,coords)
        ### After this the whole image is on BG (img is unused anymore) 
        
        ################
        #### shadow ####
        ################
        if Shadow:
            Arr = np.array(BG)
            
            c = random.randint(int(Arr[:,:,0].shape[0]/5),int(Arr[:,:,0].shape[0]/1.2))
            shadow = np.ones((Arr[:,:,0].shape[0],Arr[:,:,0].shape[1]))
            for i in range (c,Arr[:,:,0].shape[0]):
                shadow[i,:] = 0

            shadow = shadow * DARKNESS
            Arr[:,:,0] = np.clip(Arr[:,:,0] + shadow, a_min = 0, a_max = 255) 
            Arr[:,:,1] = np.clip(Arr[:,:,1] + shadow, a_min = 0, a_max = 255) 
            Arr[:,:,2] = np.clip(Arr[:,:,2] + shadow, a_min = 0, a_max = 255) 

            CarPlate_img = Image.fromarray(Arr)
        else:
            CarPlate_img = BG
        #######################################
        # Apply mean filter to blur the image #
        #######################################
        if blur:
            kernel_size = blur_kernel_size
            kernel = np.ones((kernel_size,kernel_size))
            kernel = ImageFilter.Kernel(kernel.shape, kernel = kernel.flatten(), scale=kernel_size**2 , offset=0)
            CarPlate_img = CarPlate_img.filter(kernel)#Apply mean filter

        ####################
        ## Apply rotation ##
        ####################
        
        tmp_W, tmp_H = CarPlate_img.size
        CarPlate_img = CarPlate_img.rotate(angle_to_rotate)
        draw = ImageDraw.Draw(CarPlate_img)

        for i in coords:
            x1, y1 = i[0]
            x2, y2 = i[1]
            x_1, y_1 = rotate_point((tmp_W/2,tmp_H/2), (x1, y1), angle_to_rotate * math.pi / 180)
            x_2, y_2 = rotate_point((tmp_W/2,tmp_H/2), (x1, y2), angle_to_rotate * math.pi / 180)
            x_3, y_3 = rotate_point((tmp_W/2,tmp_H/2), (x2, y1), angle_to_rotate * math.pi / 180)
            x_4, y_4 = rotate_point((tmp_W/2,tmp_H/2), (x2, y2), angle_to_rotate * math.pi / 180)
            x1 = int(min([x_1,x_2,x_3,x_4]))
            x2 = int(max([x_1,x_2,x_3,x_4]))
            y1 = int(min([y_1,y_2,y_3,y_4]))
            y2 = int(max([y_1,y_2,y_3,y_4]))
            coords[coords.index(i)] = (x1, y1, x2, y2)
            #draw.rectangle((x1,y1,x2,y2), fill=None, outline=(255,255,255))
        
        ##########################
        ## Apply gaussian_noise ##
        ##########################
        if gaussian_Noise:
            Arr = np.array(CarPlate_img)
            seq = aug.Sequential([aug.AdditiveGaussianNoise(0, std_gaussian , per_channel=0.1)])#mean of noise is 0
            CarPlate_img = seq(image = Arr)
            CarPlate_img_final = Image.fromarray(CarPlate_img)

        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = '{}_{}.{}'.format(text, str(index), extension)
        elif name_format == 1:
            image_name = '{}_{}.{}'.format(str(index), text, extension)
        elif name_format == 2:
            image_name = '{}.{}'.format(str(index),extension)
        else:
            print('{} is not a valid name format. Using default.'.format(name_format))
            image_name = '{}_{}.{}'.format(text, str(index), extension)
        tmp_w , tmp_h = CarPlate_img_final.size
        coords = coordinate_resize(tmp_w, tmp_h, 600, 200, -1 ,coords)
        print(coords)
        #write xml 
        CarPlate_img_final = CarPlate_img_final.resize( (600,200), resample=0)
        xml_util.generate_xml(image_name.split('.')[0], CarPlate_img_final.size, coords, chars, out_dir)
        draw = ImageDraw.Draw(CarPlate_img_final)
        #for i in coords:
            #x1, y1 = i[0]
            #x2, y2 = i[1]
            #draw.rectangle((x1,y1,x2,y2), fill=None, outline=(255,255,255))
        # Save the image
        CarPlate_img_final.convert('RGB').save(os.path.join(out_dir, image_name))


class FakeTextDataGenerator(object):
    @classmethod
    def generate_from_tuple(cls, t):
        """
            Same as generate, but takes all parameters as one tuple
        """

        cls.generate(*t)

    @classmethod
    def generate(cls, index, text, font, out_dir, size, extension, skewing_angle, random_skew, blur, random_blur, background_type, distorsion_type, distorsion_orientation, is_handwritten, name_format, width, alignment, text_color, orientation, space_width, margins, fit):
        image = None

        margin_top, margin_left, margin_bottom, margin_right = margins
        horizontal_margin = margin_left + margin_right
        vertical_margin = margin_top + margin_bottom

        ##########################
        # Create picture of text #
        ##########################
        if is_handwritten:
            if orientation == 1:
                raise ValueError("Vertical handwritten text is unavailable")
            image = handwritten_text_generator.generate(text, text_color, fit)
        else:
            image, coords, chars = computer_text_generator.generate(text, font, text_color, size, orientation, space_width, fit)
        random_angle = random.randint(0-skewing_angle, skewing_angle)
        skewing_angle = skewing_angle if not random_skew else random_angle
        #skewing_angle = -skewing_angle
        rotated_img = image.rotate(skewing_angle, expand=1)

        #############################
        # Apply distorsion to image #
        #############################
        if distorsion_type == 0:
            distorted_img = rotated_img # Mind = blown
        elif distorsion_type == 1:
            distorted_img = distorsion_generator.sin(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        elif distorsion_type == 2:
            distorted_img = distorsion_generator.cos(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )
        else:
            distorted_img = distorsion_generator.random(
                rotated_img,
                vertical=(distorsion_orientation == 0 or distorsion_orientation == 2),
                horizontal=(distorsion_orientation == 1 or distorsion_orientation == 2)
            )

        ##################################
        # Resize image to desired format #
        ##################################

        # Horizontal text
        if orientation == 0:
            new_width = int(distorted_img.size[0] * (float(size - vertical_margin) / float(distorted_img.size[1])))
            resized_img = distorted_img.resize((new_width, size - vertical_margin), Image.ANTIALIAS)
            background_width = width if width > 0 else new_width + horizontal_margin
            background_height = size
        # Vertical text
        elif orientation == 1:
            new_height = int(float(distorted_img.size[1]) * (float(size - horizontal_margin) / float(distorted_img.size[0])))
            resized_img = distorted_img.resize((size - horizontal_margin, new_height), Image.ANTIALIAS)
            background_width = size
            background_height = new_height + vertical_margin
        else:
            raise ValueError("Invalid orientation")

        #############################
        # Generate background image #
        #############################
        if background_type == 0:
            background = background_generator.gaussian_noise(background_height, background_width)
        elif background_type == 1:
            background = background_generator.plain_white(background_height, background_width)
        elif background_type == 2:
            background = background_generator.quasicrystal(background_height, background_width)
        else:
            background = background_generator.picture(background_height, background_width)

        #############################
        # Place text with alignment #
        #############################

        new_text_width, _ = resized_img.size

        if alignment == 0 or width == -1:
            background.paste(resized_img, (margin_left, margin_top), resized_img)
        elif alignment == 1:
            background.paste(resized_img, (int(background_width / 2 - new_text_width / 2), margin_top), resized_img)
        else:
            background.paste(resized_img, (background_width - new_text_width - margin_right, margin_top), resized_img)


        # Calculate bounding box after resize and rotate
        new_coords = []

        w_percentage = resized_img.size[0] / distorted_img.size[0]
        h_percentage = resized_img.size[1] / distorted_img.size[1]

        for coord in coords:
            xmin = coord[0][0]
            ymin = coord[0][1]
            xmax = coord[1][0]
            ymax = coord[1][1]

            xmin = xmin * w_percentage
            ymin = ymin * h_percentage
            xmax = xmax * w_percentage
            ymax = ymax * h_percentage

            oh = resized_img.size[1]/ 2 
            ow = resized_img.size[0]/ 2
            if skewing_angle > 0: 
                origin = ow + .45 * resized_img.size[0], oh - .45 * resized_img.size[1]
            elif skewing_angle < 0:
                 origin = ow - .45 * resized_img.size[0], oh + .45 * resized_img.size[1]   
            else:
                origin = oh, ow     

            angel =  -math.radians(skewing_angle) 

            x1, y1 = rotate_point(origin, (xmin, ymin), angel)
            x2, y2 = rotate_point(origin, (xmin, ymax), angel)
            x3, y3 = rotate_point(origin, (xmax, ymax), angel)
            x4, y4 = rotate_point(origin, (xmax, ymin), angel)

            xmin = min([x1, x2, x3, x4])
            ymin = min([y1, y2, y3, y4])
            xmax = max([x1, x2, x3, x4])
            ymax = max([y1, y2, y3, y4])

            if skewing_angle > 0:
                xmax = xmax + .5*angel*(xmax-xmin)
                ymax = ymax - .5*angel*(ymax-ymin)
            elif skewing_angle < 0:
                xmin = xmin - .5*angel*(xmax-xmin)
                xmax = xmax - 1.3*angel*(xmax-xmin)
                ymin = ymin + .3*angel*(ymax-ymin)

            #adjust to margins
            xmin, ymin, xmax, ymax = xmin + margin_left, ymin + margin_top, xmax + margin_left, ymax + margin_top

            new_coords.append([(xmin, ymin), (xmax, ymax)])
            drawer = ImageDraw.Draw(background)
            drawer.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)], fill = (0, 255, 0), width=1)    

        ##################################
        # Apply gaussian blur #
        ##################################

        final_image = background.filter(
            ImageFilter.GaussianBlur(
                radius=(blur if not random_blur else random.randint(0, blur))
            )
        )

        #####################################
        # Generate name for resulting image #
        #####################################
        if name_format == 0:
            image_name = '{}_{}.{}'.format(text, str(index), extension)
        elif name_format == 1:
            image_name = '{}_{}.{}'.format(str(index), text, extension)
        elif name_format == 2:
            image_name = '{}.{}'.format(str(index),extension)
        else:
            print('{} is not a valid name format. Using default.'.format(name_format))
            image_name = '{}_{}.{}'.format(text, str(index), extension)

        
        #write xml 
        xml_util.generate_xml(image_name.split('.')[0], final_image.size, new_coords, chars, out_dir)    

        # Save the image
        final_image.convert('RGB').save(os.path.join(out_dir, image_name))

        
    
def rotate_point(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) + math.sin(angle) * (py - oy)
    qy = oy - math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def coordinate_resize(old_w, old_h, new_w, new_h, frame_number ,coordinates):
    th_w = new_w/old_w
    th_h = new_h/old_h
    if frame_number == 0:
        trans_w = 34
        trans_h = 20
    elif frame_number == 1:
        trans_w = 15
        trans_h = 17
    elif frame_number == 2:
        trans_w = 13
        trans_h = 9
    elif frame_number == 3:
        trans_w = 92
        trans_h = 35
    elif frame_number == 4:
        trans_w = 4
        trans_h = 4
    elif frame_number == 5:
        trans_w = 70
        trans_h = 57
    elif frame_number == 6:
        trans_w = 5
        trans_h = 5
    elif frame_number == -1:
        trans_w = 0
        trans_h = 0
    else:
        raise Exception("Wrong Frame_number")
    
    for i in range (len(coordinates)):
        (x_min, y_min, x_max, y_max) = coordinates[i]
        x_min = x_min * th_w + trans_w
        x_max = x_max * th_w + trans_w
        y_min = y_min * th_h + trans_h
        y_max = y_max * th_h + trans_h
        coordinates[i] = [(x_min, y_min), (x_max, y_max)]
    return coordinates
    
