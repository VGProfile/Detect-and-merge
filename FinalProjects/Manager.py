import math
import random
from FinalProjects.Img_processing import Img_processing as ipr
from FinalProjects.Object_process import Object_process as od
import numpy as np
import cv2

def generate_image_for_processing(size,img):
    Img_process = ipr(size, size)
    img_for_procces = img

    w = (random.randint(5, 8) / 10)
    h = (random.randint(8, 10) / 10)
    angle = random.randint(10, 30)


    img_B_s = Img_process.resize_by_points(img, 150, 150)
    shift_B = od.rectangle_area_of_object_points(Img_process.get_points_by_image(
        od.get_image_without_background_color(img_B_s, (255, 255, 255))))

    # Merge and brush by randomize form of the image
    img_B = ipr(size, size)
    img_new = img_B.create_blank()

    coord_b = ipr.get_points_by_image(od.get_image_without_background_color(img_B_s, (255, 255, 255)))
    img_new = Img_process.img_brush_by_count_of_points(img_new, coord_b, (255, 100, 100), 150 - int(shift_B[0][1]),2000)

    new_size_img2 = ipr.resize(img_new, w, h)
    bg = Img_process.create_blank()
    new_red_image_by_rotation = ipr.rotate_image(new_size_img2, angle)
    pt = ipr.get_points_by_image(od.get_image_without_background(new_red_image_by_rotation))
    new_image_for_stack = ipr.img_brush_by_count_of_points(bg, pt, (0, 0, 255), 0, 1200)

    return np.hstack((img_new, new_image_for_stack))

def find_object_and_merge(image,output_size):
    temp = image.copy()
    dt = od()
    figures = dt.detection(image)
    cv2.imshow("DetectImg", image)

    # find orders of the figures
    img_frs = od.cut_image(temp, figures[0].points)
    img_sec = od.cut_image(temp, figures[1].points)
    img_red = None
    img_blue = None
    figure_red = None
    figures_blue = None
    if (od.consist_colors(img_frs, (0, 0, 255)) > 0):
        img_red = img_frs
        figure_red = figures[0]
        img_blue = img_sec
        figures_blue = figures[1]
    else:
        img_blue = img_frs
        figure_blue = figures[0]
        img_red = img_sec
        figures_red = figures[1]

    rel_angle = od.relative_angle(figures[0].angle, figures[1].angle)

    # Normalize figures size
    test = ipr.rotate_image(img_red, (figure_red.normale))
    resize = test.copy()
    t_f = dt.detection(test)
    reb = ipr.cut_image(resize, t_f[0].points)

    h, w = img_blue.shape[:2]
    oldH, oldW = reb.shape[:2]
    final_red = ipr.resize(reb, w / oldW, h / oldH)
    cv2.imshow("final", final_red)

    # find coordinates for shift
    coord = od.get_image_without_background(final_red)
    shift = od.rectangle_area_of_object_points(ipr.get_points_by_image(coord))
    coord_s = od.get_image_without_background(img_blue)
    shift_s = od.rectangle_area_of_object_points(ipr.get_points_by_image(coord_s))

    # Merge and brush normalize forms
    img_n = ipr(output_size, output_size)
    img_new = img_n.create_blank()

    coord_r = ipr.get_points_by_image(coord)
    coord_b = ipr.get_points_by_image(coord_s)
    img_frs = ipr.img_brush_by_points(img_new, coord_r, (0, 0, 255), 150 - int(shift[0][1]))
    img_final = ipr.img_brush_by_points(img_frs, coord_b, (255, 0, 0), 150 - int(shift[0][1]))
    return img_final

input_image = generate_image_for_processing(400,cv2.imread("w512h5121390845821B512.png"))
merge_image = find_object_and_merge(input_image,400)
final = np.hstack((input_image,merge_image))
cv2.imshow("FinalImg", final)
cv2.waitKey(0)