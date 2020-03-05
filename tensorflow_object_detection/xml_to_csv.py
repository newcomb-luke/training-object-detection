import xml.etree.ElementTree as ET
import os

def xml_to_dict(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    member = root.find('object')

    # (name, xmin, ymin, xmax, ymax)
    value = (root.find('filename').text, member[0].text, member[4][0].text, member[4][1].text, member[4][2].text, member[4][3].text)

    value = {
        'filename' : value[0],
        'object' : value[1],
        'xmin' : value[2],
        'ymin' : value[3],
        'xmax' : value[4],
        'ymax' : value[5]
    }

    return value

def read_xmls(directory_path):
    
    files = os.listdir(directory_path)

    data = []

    for i, filename in enumerate(files):

        file_name = filename.split('.')[0]
        
        extension = filename.split('.')[1]

        print(file_name)

        if extension == 'xml':
            file_path = os.path.join(directory_path, filename)

            data.append(xml_to_dict(file_path))
    
    return data

def sort_label_list(label_list):
    filename_list = [None] * len(label_list)
    labels_sorted = [None] * len(label_list)

    length = len(label_list)

    for i in range(length):
        filename_list[i] = label_list[i]['filename']
    
    filenames_sorted = sorted(filename_list)

    for i in range(length):
        labels_sorted[i] = label_list[filename_list.index(filenames_sorted[i])]
    
    return labels_sorted

def export_as_csv(sorted_labels, csv_path):

    with open(csv_path, 'w') as fp:
        for label in sorted_labels:
            for key, value in label.items():
                fp.write('{},'.format(value))
            fp.write('\n')

def main():
    data = read_xmls('/home/luke/Documents/git-repositories/training-object-detection/images/test/target/')

    sorted_labels = sort_label_list(data)

    export_as_csv(sorted_labels, 'test-labels.csv')

if __name__ == '__main__':
    main()
