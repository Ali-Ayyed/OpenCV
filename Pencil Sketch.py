
import cv2

cap =  cv2.VideoCapture(0)
# img = cv2.imread('')


while True:
  suc,frame = cap.read()
  img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  img_invert = cv2.bitwise_not(img_gray)
  blurred = cv2.GaussianBlur(img_invert, (23,23), 0)
  invert = cv2.bitwise_not(blurred)
  pencil_sketch = cv2.divide(img_gray, invert, scale=250.0)
  cv2.imshow("Sketch", pencil_sketch)
  if cv2.waitKey(21) & 0xFF == ord('q'):
    break











