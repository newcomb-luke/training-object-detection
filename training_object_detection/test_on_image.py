import cv2

BASE_DIR = '/home/luke/Documents/git-repositories/training-object-detection/'

def main():
    tensorflowNet = cv2.dnn.readNetFromTensorflow(BASE_DIR + 'new_graph.pb')

    img = cv2.imread(BASE_DIR + 'test.png')
    rows, cols, channels = img.shape

    tensorflowNet.setInput(cv2.dnn.blobFromImage(img, size=(200, 200), swapRB=True, crop=False))

    networkOutput = tensorflowNet.forward()

    for detection in networkOutput[0,0]:
        score = float(detection[2])
        if score > 0.2:
            left = detection[3] * cols
            top = detection[4] * rows
            right = detection[5] * cols
            bottom = detection[6] * rows

            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)

    cv2.imshow('Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()