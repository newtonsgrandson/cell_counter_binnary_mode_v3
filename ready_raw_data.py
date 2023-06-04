import cv2
import numpy as np
from sel_rectangle import select_rectangle
import os

def crop_and_save(image_path, coords, output_path):
    # Read the original image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale

    # Crop the image using numpy slicing and the provided coordinates
    cropped_img = img[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]]

    _, binary = cv2.threshold(cropped_img, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(contours)

    # Loop through contours and draw bounding rectangles around the white clusters
    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle on the original image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Save the cropped image
    cv2.imwrite(output_path, binary_img)

def find_white_clusters(img):

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to isolate white regions.
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours and draw bounding rectangles around the white clusters
    for contour in contours:
        # Get bounding box coordinates
        x, y, w, h = cv2.boundingRect(contour)

        # Draw rectangle on the original image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the image

def main():

    list_dir = list(os.listdir("raw_data"))

    try:
        os.mkdir("ready_raw_data")
    except:
        pass
    
    output_path = "ready_raw_data"
    for i in range(list_dir.__len__()):
        print(list_dir[i])
        coords = select_rectangle("raw_data/" + list_dir[i])        
        print(str(output_path) + "/" + str(i))
        crop_and_save("raw_data/" + list_dir[i], coords, str(output_path) + "/" + str(i) + ".jpg")

if __name__ == "__main__":
    main()