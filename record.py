import cv2

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 170)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)
for i in range(45):
    return_value, image = camera.read()
    cv2.imwrite('recording/00'+str(i)+'.png', image)
    cv2.imshow('Frame', image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
del(camera)