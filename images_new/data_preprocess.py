import os
import random
import shutil

# BASE_DIR = '/home/luke/Documents/git-repos/training-object-detection/'

BASE_DIR = '/home/luke/Documents/git-repositories/training-object-detection/'

TARGET_IMAGES_DIR =  BASE_DIR + 'training_set/target/'
NOTARGET_IMAGES_DIR =  BASE_DIR + 'training_set/no_target/'
TRAIN_IMAGES_DIR = BASE_DIR + 'training_set/train/'
TEST_IMAGES_DIR = BASE_DIR + 'training_set/valid/'

TRAIN_PERCENT = 75

def main():
    
    print('Reading files')

    train_target_dir = TRAIN_IMAGES_DIR + 'target/'
    test_target_dir = TEST_IMAGES_DIR + 'target/'

    train_notarget_dir = TRAIN_IMAGES_DIR + 'no_target/'
    test_notarget_dir = TEST_IMAGES_DIR + 'no_target/'

    if not os.path.exists(train_target_dir):
        os.makedirs(train_target_dir)
    
    if not os.path.exists(test_target_dir):
        os.makedirs(test_target_dir)
    
    if not os.path.exists(train_notarget_dir):
        os.makedirs(train_notarget_dir)
    
    if not os.path.exists(test_notarget_dir):
        os.makedirs(test_notarget_dir)

    target_files = os.listdir(TARGET_IMAGES_DIR)
    notarget_files = os.listdir(NOTARGET_IMAGES_DIR)

    print('Sorting files')

    target_files_sorted = sorted(target_files)
    notarget_files_sorted = sorted(notarget_files)

    random.shuffle(target_files_sorted)
    random.shuffle(notarget_files_sorted)

    num_train_target = int( len(target_files_sorted) * ( TRAIN_PERCENT / 100 ) )
    num_train_notarget = int( len(notarget_files_sorted) * ( TRAIN_PERCENT / 100 ) )
    num_test_target = len(target_files_sorted) - num_train_target
    num_test_notarget = len(notarget_files_sorted) - num_train_notarget

    train_images_target = [[None] for _ in range(num_train_target)]
    train_images_notarget = [[None] for _ in range(num_train_notarget)]
    test_images_target = [[None] for _ in range(num_test_target)]
    test_images_notarget = [[None] for _ in range(num_test_notarget)]
    
    for i in range(num_train_target):
        train_images_target[i] = target_files_sorted[i]

    for i in range(num_train_notarget):
        train_images_notarget[i] = notarget_files_sorted[i]
    
    for i in range(num_test_target):
        test_images_target[i] = target_files_sorted[i + num_train_target]

    for i in range(num_test_notarget):
        test_images_notarget[i] = notarget_files_sorted[i + num_train_notarget]

    print('Copying files')

    for i, file_path in enumerate(train_images_target):
        file_ext = file_path.split('.')[1]
        file_name = 'target{:04}.{}'.format(i, file_ext)
        source = os.path.join( TARGET_IMAGES_DIR, file_path )
        destination = os.path.join( train_target_dir, file_name )
        shutil.copyfile(source, destination)
    
    for i, file_path in enumerate(test_images_target):
        file_ext = file_path.split('.')[1]
        file_name = 'target{:04}.{}'.format(i, file_ext)
        source = os.path.join( TARGET_IMAGES_DIR, file_path )
        destination = os.path.join( test_target_dir, file_name )
        shutil.copyfile(source, destination)
    
    for i, file_path in enumerate(train_images_notarget):
        file_ext = file_path.split('.')[1]
        file_name = 'no_target{:04}.{}'.format(i, file_ext)
        source = os.path.join( NOTARGET_IMAGES_DIR, file_path )
        destination = os.path.join( train_notarget_dir, file_name )
        shutil.copyfile(source, destination)
    
    for i, file_path in enumerate(test_images_notarget):
        file_ext = file_path.split('.')[1]
        file_name = 'no_target{:04}.{}'.format(i, file_ext)
        source = os.path.join( NOTARGET_IMAGES_DIR, file_path )
        destination = os.path.join( test_notarget_dir, file_name )
        shutil.copyfile(source, destination)

    print('Done.')

if __name__ == "__main__":
    main()