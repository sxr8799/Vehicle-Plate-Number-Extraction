# Vehicle Plate Number Extraction

## Overview:
This is an implementation of a Vehicle Number Extraction program written in Python that extracts from an image the vehicle license plate number data using NumPy, imutils, easyOCR, OpenCV(cv2) and matplotlib. The program works by first prompting the user to specify the location of the image to be used. After which, the image is read in, grayscaled and blurred. The second step is applying filters and specifying the edges for localization. The third step will be finding the contour and applying masking to the image. Finally, Easyocr will be used to read the text, after which the program will render the results using matplotlib.
