import cv2

#importing the classifiers
faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade=cv2.CascadeClassifier('haarcascade_eye.xml')
noseCascade=cv2.CascadeClassifier('Nariz.xml')
mouthCascade=cv2.CascadeClassifier('Mouth.xml')

# function to plot the boundary
def plot_edges(image, classifier, scaleFactor, minNeighbors, color, text):
    gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image,scaleFactor,minNeighbors)
    coords=[]
    for (x, y, w, h) in features:
        cv2.rectangle(image,(x,y),(x+w, y+h),color,2)
        cv2.putText(image, text, (x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
        coords=[x, y, w, h]
    return coords
# function to detect the face and its features
def Detect_Face(image,faceCascade,eyeCascade,noseCascade,mouthCascade):
    color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
    coords = plot_edges(image, faceCascade, 1.1, 10, color['blue'],"Face")

    if len(coords)==4:
         # Updating region of interest by cropping image
        roi_img = image[coords[1]:coords[1]+coords[3], coords[0]:coords[0]+coords[2]]
        # Passing roi, classifier, scaling factor, Minimum neighbours, color, label text
        coords = plot_edges(roi_img, eyeCascade, 1.1, 12, color['red'], "Eye")
        coords = plot_edges(roi_img, noseCascade, 1.1, 4, color['green'], "Nose")
        coords = plot_edges(roi_img, mouthCascade, 1.1, 20, color['white'], "Mouth")
    return image


#initialising webcam.
video = cv2.VideoCapture(0)

# while loop to capture video feed frame by frame
while True:
    ret, image = video.read() # here, image is the frame
    image = Detect_Face(image,faceCascade,eyeCascade)
    cv2.imshow("Face detection",image)
    # key=cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'): # press q to quit program
        break

# releasing the video camera instance
video.release()

#closing the video camera window
cv2.destroyAllWindows()
