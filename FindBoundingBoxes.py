import numpy as np
import cv2


def Find_BB(image,image_number) :
    # Load image, grayscale, median blur, sharpen image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 100, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    close = cv2.bitwise_not(close)

    # Find contours and filter using threshold area
    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    min_area = 2
    max_area = 5000
    BB= []
    maxFoundArea = 0
    for c in cnts:
        area = cv2.contourArea(c)
        maxFoundArea = max(maxFoundArea,area)

    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area and area > maxFoundArea/2:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            BB.append((x + w/2, y + h/2, w, h))

    cv2.imwrite('./Images3/screenshot_{}.png'.format(image_number), image)
    return BB


def draw_bounding_boxes_on_rgb_image(rgb_image, bounding_boxes):

    # Convert the RGB image to BGR format for use with OpenCV
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    # Draw each bounding box on the image
    for (x_center, y_center, width, heigth) in bounding_boxes:
        cv2.rectangle(bgr_image, (int(x_center-width/2), int(y_center-heigth/2)), (int(x_center+width/2), int(y_center+heigth/2)), (255, 0, 0), 2)
        cv2.putText(bgr_image, f"({x_center}, {y_center})", (int(x_center-width/2), int(y_center-heigth/2) -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)

    # Convert the BGR image back to RGB format
    result_rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    return result_rgb_image

def PositionOfBox_RelativeToCenter(BoxBoundaries, factor=0.5) :
    Coords = []
    for elem in BoxBoundaries :
        x = (elem[0] / factor) + 300
        y = (elem[1] / factor) + 200
        Coords.append((x-960,y-540))
    return(Coords)


def mouse_movement_required_xy(target_displacement_x, target_displacement_y,
                               screen_width=1920, screen_height=1080,
                               mouse_move_for_half_screen=405*1.34, mouse_move_for_half_height=275*1.12):

    fraction_of_screen_x = target_displacement_x / (screen_width / 2)
    mouse_move_x = fraction_of_screen_x * mouse_move_for_half_screen

    fraction_of_screen_y = target_displacement_y / (screen_height / 2)
    mouse_move_y = fraction_of_screen_y * mouse_move_for_half_height

    return int(mouse_move_x), int(mouse_move_y)