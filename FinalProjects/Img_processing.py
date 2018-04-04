import math
import random
import numpy as np
from random import randrange
import imutils
import cv2
import numpy.linalg as la

class Img_processing:

    def __init__(self, blank_size_wight, blank_size_height):
        self.blank_size_height = blank_size_height
        self.blank_size_wight = blank_size_wight

    @staticmethod
    def img_brush_by_points(img, points, color, shift):
        for elem in points:
            img[elem[1] + shift, elem[0] + shift] = color
        return img

    @staticmethod
    def img_brush_by_count_of_points(img, points, color, shift, count):

        area_of_image = points

        while (count > 0):
            random_index = randrange(0, len(area_of_image))
            x, y = area_of_image[random_index]
            img[y+shift][x+shift] = color
            count = count - 1
        return img

    @staticmethod
    def resize(image, width_muilt, height_muilt):
        height, width = image.shape[:2]
        return cv2.resize(image, (int(width_muilt * width), int(height_muilt * height)), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def resize_by_points(image, width, height):
        return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def rotate_image(image, angle):
        # Default rotation of image...
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        return cv2.warpAffine(image, M, (nW, nH))

    @staticmethod
    def cut_image(image, points):
        np_arr = np.array(points)

        # Crop the bounding rectangle
        rectangle = cv2.boundingRect(np_arr)
        alpha, beta, width, height = rectangle
        croped = image[beta:beta + height, alpha:alpha + width].copy()

        # Make mask
        np_arr = np_arr - np_arr.min(axis=0)
        mask = np.zeros(croped.shape[:2], np.uint8)
        cv2.drawContours(mask, [np_arr], -1, (100, 100, 100), -1, cv2.LINE_AA)

        cv2.imshow("crosddsd.png", mask)

        # Do bit-op
        image = cv2.bitwise_and(croped, croped, mask=mask)

        # Add the white background
        # background = np.ones_like(croped, np.uint8) * 255
        # cv2.bitwise_not(background, background, mask=mask)
        # image = background + distance

        cv2.imshow("croped.png", image)

        return image


    def create_blank(self):
        # Create black blank image
        image = np.zeros((self.blank_size_height, self.blank_size_wight, 3), np.uint8)
        # Since OpenCV uses BGR, convert the color first
        color = tuple(reversed((0, 0, 0)))
        # Fill image with color
        image[:] = color
        return image

    def create_image(self, image, shift_area,width, height,
                     color, count_of_cells):
        # Calculating area for image generation
        area_of_image = []
        for y in range (0+shift_area,height+shift_area):
            for x in range(0+shift_area,width+shift_area):
                if(x < 40+shift_area):
                    area_of_image.append((y,x))
                elif(y< 30+shift_area and x<120+shift_area):
                    area_of_image.append((y,x))

        # Set priority points


        # Implementation of the operation of the sprayer points
        while (count_of_cells > 0):
            random_index = randrange(0, len(area_of_image))
            x, y = area_of_image[random_index]
            image[x][y] = color
            count_of_cells = count_of_cells - 1
        return image

    # Support function
    @staticmethod
    def get_points_by_image(image):
        points = []
        for num in range(0, len(image[0])):
            points.append((image[1][num], image[0][num]))
        return points

    def points_of_rotation_func(self, coord, angle, center=(0, 0)):
        points = self.get_points_by_image(coord)
        points_trans = []
        center_x, center_y = center

        # Get points relative to the angle
        for elem in points:
            x = int(((elem[0]-center_x) * np.math.cos(angle)) + ((elem[1]-center_y) * np.math.sin(angle)))
            y = int(((elem[0]-center_x) * -np.math.sin(angle)) + ((elem[1]-center_y) * np.math.cos(angle)))
            points_trans.append((x, y))
        return points_trans

    def rotation_image(self, image, color,angle, center=(0, 0)):
        # Creating a new image to rotate
        img_new = self.create_blank()

        # Get the points of the object color
        coord = np.where(np.all(image == color, axis=-1))

        # Painting points relative to the new position
        rotated_points = self.points_of_rotation_func(coord,np.math.radians(angle), center)
        for elem in rotated_points:
            img_new[elem[0],elem[1]] = color
        return img_new

    def points_of_scaling_func(self, coord, width, height, center=(0, 0)):
        points = self.get_points_by_image(coord)
        points_trans = []
        center_x, center_y = center

        # Get points relative to the angle
        for elem in points:
            x = int(elem[0]*width)
            y = int(elem[1]*height)
            points_trans.append((x, y))
        return points_trans

    def scaling_image(self, image, color, width, height):
        # Creating a new image to rotate
        img_new = self.create_blank()

        # Get the points of the object color
        coord = np.where(np.all(image == color, axis=-1))

        # Painting points relative to the new position
        rotated_points = self.points_of_scaling_func(coord, width,height)
        for elem in rotated_points:
            img_new[elem[0],elem[1]] = color
        return img_new


