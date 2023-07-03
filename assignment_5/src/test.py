import cv2
img = cv2.imread('./img/Art/view1.png')
x,y,w,h = cv2.selectROI("location", img, False)
print(f"x: {x}, y: {y}")
print(f"w: {w}, h: {h}")
cv2.destroyAllWindows()