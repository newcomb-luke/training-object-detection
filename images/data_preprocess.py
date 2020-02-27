import os
import random
import shutil

BASE_DIR = '/home/luke/Documents/git-repos/training-object-detection/'

ALL_IMAGES_DIR =  BASE_DIR + 'images/all/'
TRAIN_IMAGES_DIR = BASE_DIR + 'images/train/'
TEST_IMAGES_DIR = BASE_DIR + 'images/test/'

TRAIN_PERCENT = 60

def main():
    
    print('Reading files')

    train_target_dir = TRAIN_IMAGES_DIR + 'target/'
    test_target_dir = TEST_IMAGES_DIR + 'target/'

    if not os.path.exists(train_target_dir):
        os.makedirs(train_target_dir)
    
    if not os.path.exists(test_target_dir):
        os.makedirs(test_target_dir)

    all_files = os.listdir(ALL_IMAGES_DIR)

    print('Sorting files')

    all_files_sorted = sorted(all_files)

    num_images = int(len(all_files_sorted) / 2)

    all_images = [[[None] for _ in range(2)] for _ in range(num_images)]

    for i in range(num_images):
        all_images[i][0] = all_files_sorted[i * 2]
        all_images[i][1] = all_files_sorted[(i * 2) + 1]

    random.shuffle(all_images)
    random.shuffle(all_images)

    num_train = int( num_images * ( TRAIN_PERCENT / 100 ) )
    num_test = num_images - num_train

    train_images = [[[None] for _ in range(2)] for _ in range(num_train)]
    test_images = [[[None] for _ in range(2)] for _ in range(num_test)]
    
    for i in range(num_train):
        train_images[i] = all_images[i]
    
    for i in range(num_test):
        test_images[i] = all_images[i + num_train]

    print('Copying files')

    for i, image in enumerate(train_images):
        for file_path in image:
            file_ext = file_path.split('.')[1]
            file_name = 'target{:04}.{}'.format(i, file_ext)
            source = os.path.join( ALL_IMAGES_DIR, file_path )
            destination = os.path.join( train_target_dir, file_name )
            shutil.copyfile(source, destination)
    
    for i, image in enumerate(test_images):
        for file_path in image:
            file_ext = file_path.split('.')[1]
            file_name = 'target{:04}.{}'.format(i, file_ext)
            source = os.path.join( ALL_IMAGES_DIR, file_path )
            destination = os.path.join( test_target_dir, file_name )
            shutil.copyfile(source, destination)
    
    print('Done.')

if __name__ == "__main__":
    main()