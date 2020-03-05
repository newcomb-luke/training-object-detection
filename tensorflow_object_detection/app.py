import cv2

def main():
    tensorflowNet = cv2.dnn.readNetFromTensorflow('/home/luke/Documents/git-repositories/training-object-detection/target_detector/transformed_frozen_inference_graph.pb',
                                                  '/home/luke/Documents/git-repositories/training-object-detection/target_detector/ssd_mobilenet_v1_target_2020_03_03.pbtxt')
    img = cv2.imread('/home/luke/Documents/git-repositories/training-object-detection/test.png')

    rows, cols, channels = img.shape

    tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))

    networkOutput = tensorflowNet.forward()

    for detection in networkOutput[0,0]:

        score = float(detection[2])

        if score > 0.2:
            
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows
    
            #draw a red rectangle around detected objects
            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
 
    # Show the image with a rectagle surrounding the detected objects 
    cv2.imshow('Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

