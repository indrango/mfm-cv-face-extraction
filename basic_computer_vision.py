import cv2
import numpy as np

# read image
img = cv2.imread('images/Profile.png')
# img = cv2.imread('images/2-people.jpeg')
print(img.shape)
print(type(img))

# resize image 50%
reshape_img = cv2.resize(img, (img.shape[1]/2, img.shape[0]/2))
print(reshape_img.shape)

# # display rectangle / bounding box
# cv2.rectangle(image, (left_top_coordinate), (right_bottom_coordinate), (color), line_width)
cv2.rectangle(reshape_img, (120, 200), (210, 300), (255, 255, 0), 2)

x1 = 40
y1 = 160
cv2.rectangle(reshape_img, (x1, y1), (x1+250, y1+400), (255, 0, 0), 3)

# display text
# cv2.putText(image, text, bottom_left_corner, font, font_scale, font_color, line_type)
cv2.putText(reshape_img, 'Indra Nugraha', (100, 140),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# display image
cv2.imshow('Image', reshape_img)
cv2.waitKey(0)

# write image
# cv2.imwrite('image_path', image)
cv2.imwrite('images/new_profile.png', reshape_img)

