import os
import random
import math

from PIL import Image, ImageFilter, ImageDraw

import computer_text_generator
import background_generator
import distorsion_generator
import xml_util
try:
    import handwritten_text_generator
except ImportError as e:
    print('Missing modules for handwritten text generation.')


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

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

    
