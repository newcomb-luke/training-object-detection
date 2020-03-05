import os
import cv2

# BASE_DIR = '/home/luke/Documents/git-repos/training-object-detection/'

BASE_DIR = '/home/luke/Documents/git-repositories/training-object-detection/'

TARGET_IMAGES_DIR =  BASE_DIR + 'images_new/target/'
NOTARGET_IMAGES_DIR =  BASE_DIR + 'images_new/no_target/'

NOTARGETS_CROPPED_DIR = BASE_DIR + 'cropped_notargets/'

def main():
    notarget_files = os.listdir(NOTARGET_IMAGES_DIR)
    notarget_files_sorted = sorted(notarget_files)

    for notarget_image in notarget_files_sorted:
        img = cv2.imread(NOTARGET_IMAGES_DIR + notarget_image)
        for x in range(4):
            for y in range(4):
                yc = y * 45
                xc = x * 45
                crop_img = img[y:y+45, x:x+45]
                new_filename = notarget_image.split('.')[0] + '{}{}.png'.format(x, y)
                cv2.imwrite(NOTARGETS_CROPPED_DIR + new_filename, crop_img)

if __name__ == '__main__':
    main()