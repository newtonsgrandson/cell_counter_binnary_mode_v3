import cv2
import numpy as np

def select_rectangle(image_path):
    # Original image
    orig_img = cv2.imread(image_path)

    target_size=(int(orig_img[0].__len__() / 3), int(orig_img.__len__() / 3))
    # Calculate scale based on original size and target size
    scale_y = orig_img.shape[0] / target_size[0]
    scale_x = orig_img.shape[1] / target_size[1]

    # Create a window
    windowName = 'Draw Rectangle'
    cv2.namedWindow(windowName)

    # true if mouse is pressed
    drawing = False 
    (ix, iy) = (-1, -1)
    finished = False  # Added this variable to keep track of rectangle drawing state
    coords = None

    # mouse callback function
    def draw_rectangle_with_drag(event, x, y, flags, param):
        nonlocal ix, iy, drawing, img, finished, coords

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            (ix, iy) = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                img = np.copy(resized_img)
                cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), 1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            finished = True  # Set finished to True when mouse is released
            cv2.rectangle(img, (ix, iy), (x, y), (255, 0, 0), 1)
            coords = ((int(ix*3), int(iy*3)), (int(x*3), int(y*3)))

    # Bind the function to window
    cv2.setMouseCallback(windowName, draw_rectangle_with_drag)

    # Resize image
    resized_img = cv2.resize(orig_img, target_size)

    # Start with the resized image
    img = np.copy(resized_img)

    while True:
        cv2.imshow(windowName, img)
        if cv2.waitKey(1) == ord("q") or cv2.waitKey(1) == ord("Q") or finished:  # Break the loop when finished is True
            break

    cv2.destroyAllWindows()

    return coords


if __name__ == "__main__":
    select_rectangle("raw_data/1.jpg")
