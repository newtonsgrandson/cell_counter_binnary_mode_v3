import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO 

def control_coord(target_coords, control_coords):
    control_coords = list(control_coords)
    coords_del = []

    target_x_high = target_coords[0][0]
    target_x_low = target_coords[1][0]
    target_y_high = target_coords[1][1]
    target_y_low = target_coords[0][1]

    for i in range(len(control_coords)):

        control_x = control_coords[i][0]
        control_y = control_coords[i][1]

        if (control_x > target_y_high or control_x < target_y_low) or (control_y > target_x_high or control_y < target_x_low):
            coords_del.append(i)
    
    control_coords = [control_coords[i] for i in range(len(control_coords)) if not(i in coords_del)]
    return control_coords

def imshow(img):
    img = cv2.resize(img, (int(img[0].__len__() / 1), int(img.__len__() / 1)))  
    cv2.imshow("image", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def limit_vision(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, 
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
               param2 = 30, minRadius = 1220, maxRadius = 1230)
    try:
        if detected_circles == None:
            print("No detected microscope vision.")
            exit()
            
    except:
        pass

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))
    pt = detected_circles[0, :][0]
    a, b, r = pt[0], pt[1], pt[2]
    # Draw the circumference of the circle.
    cv2.circle(img, (a, b), r, (0, 255, 0), 2)

    # Draw a small circle (of radius 1) to show the center.
    cv2.circle(img, (a, b), 1, (0, 0, 255), 3)

    cropped_image = img[b-r:b+r, a-r:a+r]

    return cropped_image

def find_white_areas(img, target_coord):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    threshold_value = 242
    max_value = 255
    ret, thresholded_img = cv2.threshold(gray_img, threshold_value, max_value, cv2.THRESH_BINARY)

    # Find contours in the binary mask
    contours, hierarchy = cv2.findContours(thresholded_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # white_areas = []
    white_areas_rect = []

    cells = 0
    # Loop over the contours
    for i, contour in enumerate(contours):
        # Create a mask image for the current contour
        mask = np.zeros_like(img)
        cv2.drawContours(mask, contours, i, 255, -1)

        # Find the coordinates of the white pixels in the mask
        coords = np.column_stack(np.where(mask == 255))

        # Add the coordinates to the list of white areas
        # white_areas.append(coords)

        # Draw a rectangle around the white area
        coords = control_coord(target_coord, coords)
        length = len(coords)

        if length > 42:
            cells += 1
            x, y, w, h = cv2.boundingRect(contour)
            white_areas_rect.append([(x, y), (x+w, y+h)])
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
    return white_areas_rect, cells

def expand_white_areas(img, white_areas_rect, expand_value):

    white_areas_rect = pd.DataFrame(white_areas_rect)

    def add_value_tuple(tuple1):
        new_tuple = list()
        for i in tuple1:
            new_tuple.append(i + expand_value)
        return tuple(new_tuple)
    
    second_points = white_areas_rect.iloc[:, 1].agg(add_value_tuple)
    expand_value = -expand_value
    first_points = white_areas_rect.iloc[:, 0].agg(add_value_tuple)

    white_areas_rect = pd.concat([first_points, second_points], axis=1)

    img = np.array(img)
    white_areas = [img[i[1]:j[1], i[0]:j[0]] for i, j in white_areas_rect.values]

    return white_areas

def filter_model(white_areas, threshold):
    model_path = "yolov8m_custom.pt"
    model = YOLO(model_path)    
    class_name_dict = {0: "cell"}
    cells = 0
    for i in white_areas:
        results = model(i)[0]
        print(results)
        for result in results.boxes.data.tolist():
            print(result.boxes)
            x1, y1, x2, y2, score, class_id = result
            
    return cells




# Load the image
img = cv2.imread('sample_vision_3.jpg', cv2.IMREAD_COLOR)

# Display cropped image
img = limit_vision(img)
target_coord = [(1950, 492), (520, 1856)]
white_areas, cells = find_white_areas(img, target_coord)
control_white_areas = expand_white_areas(img, white_areas, 20)
cells = filter_model(control_white_areas, 0.5)
cells = filter_model([img], 0.5)


# Define the text and font properties
text = f"Number of Cells: {cells}"
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
thickness = 2

# Get the size of the text
size, _ = cv2.getTextSize(text, font, fontScale, thickness)

# Calculate the position of the text
height, width, channels = img.shape
x = int((width - size[0]) / 2)
y = int(height + size[1] + 10)

# Draw the text on a black background
cv2.putText(img, text, (40, 80), font, fontScale, (0, 0, 255), thickness)
scale_percent = 70 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
cv2.imwrite("Cropped Image.jpg", img)
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


# Display the resul
imshow(resized)