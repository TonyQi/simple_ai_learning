import numpy as np
import struct


minst_train_image_data_file = '../lesson10/train-images-idx3-ubyte'
minst_test_image_data_file = '../lesson10/t10k-images-idx3-ubyte'
minst_train_label_data_file = '../lesson10/train-labels-idx1-ubyte'
minst_test_label_data_file = '../lesson10/t10k-labels-idx1-ubyte'
def get_minst_train_data():
    images = read_images_file(minst_train_image_data_file)
    return images

def get_minst_test_data():
    images = read_images_file(minst_test_image_data_file)
    return images

def get_minst_train_label():
    return read_label_file(minst_train_label_data_file)
def get_minst_test_label():
    return read_label_file(minst_test_label_data_file)

def read_images_file(image_file_name):
    binfile = open(image_file_name, 'rb')
    buf = binfile.read()
    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, image_rows, image_cols = struct.unpack_from(fmt_header, buf, offset)
    print('文件名:%s,魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (image_file_name, magic_number, num_images, image_rows, image_cols))
    image_size = image_rows * image_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty((num_images, image_rows, image_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, buf, offset)).reshape((image_rows, image_cols))
        offset += struct.calcsize(fmt_image)
    return images

def read_label_file(lable_file_name):
    with open(lable_file_name, 'rb') as lbpath:
        magic, label_num = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    return labels
def get_minst_train_data_and_label():
    train_data = get_minst_train_data()
    train_label = get_minst_train_label()
    return train_data,train_label
def get_minst_test_data_and_label():
    test_data = get_minst_test_data()
    test_label = get_minst_test_label()
    return test_data,test_label