import math
import random
import sys
from FinalProjects.Figure import *
import numpy as np
from random import randrange
import imutils
import cv2
import numpy.linalg as la

class Object_process:

    @staticmethod
    def get_image_without_background(img):
        return np.where(np.any(img != (0, 0, 0), axis=-1))

    @staticmethod
    def get_image_without_background_color(img, color):
        return np.where(np.any(img != color, axis=-1))

    @staticmethod
    def img_brush_by_points(img, points, color, shift):
        for elem in points:
            img[elem[1] + shift, elem[0] + shift] = color
        return img

    @staticmethod
    def consist_colors(img, color):
        count = len(np.where(np.all(img == color, axis=-1))[0])
        print(count, "count")
        return (count > 0)

    @staticmethod
    def relative_angle(ang_first, ang_second):
        int_ang_first = int(ang_first)
        int_ang_second = int(ang_second)
        det_ang = int_ang_first - int_ang_second
        if (int_ang_second < int_ang_first):
            det_ang = -det_ang
        return det_ang

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

    @staticmethod
    def np_zip(points):
        dots = []
        for num in range(0, len(points)):
            dot = points[num][0]
            x = dot[0]
            y = dot[1]
            dots.append((x, y))
        return dots

    @staticmethod
    def distance(points):
        (begin, end) = points
        return np.sqrt((((begin[0] - end[0]) ** 2) + ((begin[1] - end[1]) ** 2)))

    @staticmethod
    def find_diagonal(points):
        max_diagonal = -sys.maxsize
        begin = None
        end = None

        for i in points:
            for j in points:
                if Object_process.distance([i, j]) > max_diagonal:
                    begin = i
                    end = j
                    max_diagonal = Object_process.distance([i, j])
        return begin, end

    @staticmethod
    def rectangle_area_of_object_points(points):
        rectangle = np.zeros((4,2), dtype="float32")
        matrix = np.matrix(points)

        # Determine the most approximate point on the diagonal,
        # where closest to the center of coordinates is the starting point

        sum = np.sum(points,axis = 1)
        rectangle[0] = points[np.argmin(sum)]
        rectangle[2] = points[np.argmax(sum)]

        # Next, the most remote points are determined relative to the opposite diagonal
        differensial = np.diff(points, axis = 1)
        rectangle[1] = points[np.argmin(differensial)]
        rectangle[3] = points[np.argmax(differensial)]
        return rectangle

    def detection(self, image):
        # ...
        Figures = []

        # ...

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 6, 3, 0.01)
        # result is dilated for marking the corners, not important
        dst = cv2.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        image[dst > 0.01 * dst.max()] = [0, 0, 255]

        # ...
        cv2.imshow("det",image)

        # To connect of independent components
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        # Get the contours of two objects
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        for c in cnts:
            # Get the key points to define the object's area
            points = self.np_zip(c)
            area = self.rectangle_area_of_object_points(points)

            # ...
            cv2.line(image, (int(area[0][0]), area[0][1]),
                            (int(area[3][0]), area[3][1]), (200, 200, 200), 1)

            normale = [[int(area[0][0]),int(area[0][1])],[int(area[3][0]), area[3][1]]]

            vector = self.find_diagonal(area)
            begin = vector[0]
            end = vector[1]
            wight, height = np.shape(image)[1::-1]
            basis = [(wight, height),(0,height)]

            cv2.line(image, (begin[0], begin[1]), (end[0], end[1]), (200, 200, 200), 1)

            # Find the angle between the normal
            cosang_n = np.dot(normale, basis)
            sinang_n = la.norm(np.cross(normale, basis))
            angle = np.degrees(np.arctan2(sinang_n, cosang_n))

            myradians_n = math.atan2(area[3][0] - area[0][0], area[3][1] - area[0][1])
            mydegrees_n = math.degrees(myradians_n)
            angle_n = round(mydegrees_n, 3)


            # TODO: To correct the support vector
            print(end[0],end[1],begin[0],begin[1],"is vector")
            if(end[1] > begin[1]):
                begin,end = end,begin


            myradians = math.atan2(begin[0] - end[0], begin[1] - end[1])
            mydegrees = math.degrees(myradians)

            # TODO: Create function for angle calculation
            cosang = np.dot(vector, basis)
            sinang = la.norm(np.cross(vector, basis))
            angle = np.degrees(np.arctan2(sinang, cosang))

            angle = 90 + round(mydegrees,3)

            text_area = str(angle_n) + ":angle ," + str(round(self.distance(vector), 4)) + ":size"
            cv2.putText(image, text_area, (int(begin[0]) - 20, int(begin[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            Figures.append(Figure(points,angle,vector,angle_n))
            # show the image

        cv2.imshow("object_detection",image)
        return Figures
