"""
Uruchomienie skryptu python preprocessing.py --i <path do zdjec> --o <path do modelu tensorflow>
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import csv
import class_list_generate
import subprocess
import os
import fnmatch
import sys, getopt
import shutil
import glob
import pandas as pd
import xml.etree.ElementTree as ET

import io
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict
from git import Repo


flags = tf.app.flags
flags.DEFINE_string('i', '', 'Path to images folder')
flags.DEFINE_string('o', '', 'Path to tensorflow')
FLAGS = flags.FLAGS


class TFModel():
    
    def clone_tfmodel(self, url_download, path_to_model):
        result = self.check_tf_dir(FLAGS.o)
        if result == False:
            Repo.clone_from(url_download, path_to_model)
        else:
            pass

    def check_tf_dir(self, path):
        if not os.path.exists(os.path.join(path, 'models-master')):
            return False
        else:
            return True

class Framework():
       
    def create_basic_folder():
        required_folders = ['training', 'data', 'images']
        path = os.path.join(os.getcwd(), 'build')
        if not os.path.exists(path):
            try:
                os.mkdir(path)
                print('---Created {}'.format(path))
            except Exception as e:
                print('Basic folder creator error - {}'.format(e))
        
        for folder in required_folders:
            pathf = os.path.join(os.getcwd(), 'build', folder)
            if not os.path.exists(pathf):
                try:
                    os.mkdir(pathf)
                    print('---Created {}'.format(pathf))
                except Exception as e:
                    print('Framwork folders creator error {}'.format(e))


    def make_framework(argv):
        
        for item in os.listdir(FLAGS.i):
            s = os.path.join(FLAGS.i, item)
            d = os.path.join(os.getcwd(), 'build', 'images', item)
            if os.path.isdir(s):
                try:
                    shutil.copytree(s, d, symlinks=False, ignore=None)
                    print('---Successfully copied images from {}'.format(inputfile))
                except:
                    pass
            else:
                shutil.copy2(s, d)

        
    def check_images_folder():
        for directory in ['test', 'train']:
            img_path = os.path.join(os.getcwd(),'build', 'images\{}'.format(directory))
            count = len([f for f in os.listdir(img_path)])
            count_xml = len([g for g in os.listdir(img_path) if fnmatch.fnmatch(g,'*.xml')])
            if count_xml != count / 2:
                print('---XML and img files aren`t equal in {}'.format(directory))
                sys.exit()
            else:
                print('---XML and img files are equal in {}'.format(directory))
                

class CSV():

    def create_labelmap(self):
        path_to_labelmap = 'build/training/labalmap.pbtxt'
        path = os.path.join(os.getcwd(), 'build', 'data', 'test_labels.csv')
        classList = class_list_generate.find_class_list(path)
        with open(path_to_labelmap, 'w+') as pbfile:
            for item in classList:
                pbfile.write("item {{ \n\tid: {} \n\tname: {}\n}}\n".format(classList.index(item)+1, item))
            print('---Successfully created labelmap file')

    def xml_to_csv(self, path):
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                        int(root.find('size')[0].text),
                        int(root.find('size')[1].text),
                        member[0].text,
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text)
                        )
                xml_list.append(value)
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df

    def make_csv(self):
        for directory in ['train', 'test']:
            image_path = os.path.join(os.getcwd(),'build', 'images/{}'.format(directory))
            xml_df = self.xml_to_csv(image_path)
            xml_df.to_csv('build/data/{}_labels.csv' .format(directory), index=None)
            print('---Successfully converted xml to csv.')


class TFRecord():

    def class_text_to_int(self, row_label):
        classList=class_list_generate.find_class_list(os.path.join(os.getcwd(), 'build', 'data', 'test_labels.csv') )
        for item in classList:
            if row_label == item:
                return classList.index(item)+1

    def split(self, df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

    def create_tf_example(self, group, path):
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(self.class_text_to_int(row['class']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    def generate_tf(self, output_file, input_csv):
        writer = tf.python_io.TFRecordWriter(os.path.join(os.getcwd(), 'build', 'data\{}'.format(output_file)))
        path = os.path.join(os.getcwd(),'build', 'images')
        examples = pd.read_csv(os.path.join(os.getcwd(), 'build','data\{}'.format(input_csv)))
        grouped = self.split(examples, 'filename')
        for group in grouped:
            tf_example = self.create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        writer.close()
        output_path = os.path.join(os.getcwd(), 'build', 'data\{}'.format(output_file) )
        print('---Successfully created the TFRecords: {}'.format(output_path))


class Tensorflow():

    def copy_to_tensorflow(self):
        path = os.path.join(os.getcwd(),'build')
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(FLAGS.o,'models\\research\\object_detection', item)
            if os.path.isdir(s):
                if not os.path.exists(d):
                    shutil.copytree(s, d, symlinks=False, ignore=None)
                    print('---Successfully copied items to {}'.format(FLAGS.o))
                else:
                    print('!!!{} exists in directory'.format(item))
            else:
                pass
                
    def query_yes_no(self, question, default='no'):
        yes = {'yes', 'ye', 'y'}
        no = {'no', 'n'}

        if default is None:
            prompt = '[y/N]'
        elif default == 'yes':
            prompt = '[y/N]'
        elif default == 'no':
            prompt = '[y/N]'
        else:
            raise ValueError('Invalid default answer {}'.format(defaul))
        while True:
            sys.stdout.write(question + prompt)
            choice = input()
            if choice in yes:
                self.start_training()
            if choice in no:
                self.exit_script()
            else:
                sys.stdout.write('Please response with y/n')

    def start_training(self):
        print('Process will start in 3...2...1...go')
        sys.exit()

    def exit_script(self):
        print('This session has ended')
        sys.exit()

def main(argv):

    framework = Framework
    csv = CSV()
    tf_record = TFRecord()
    tensorflow = Tensorflow()

    framework.create_basic_folder()
    framework.make_framework(argv)
    framework.check_images_folder()

    csv.make_csv()
    csv.create_labelmap()

    tf_record.generate_tf('train.record', 'train_labels.csv')
    tf_record.generate_tf('test.record', 'test_labels.csv')

    tensorflow.copy_to_tensorflow()
    tensorflow.query_yes_no('Do you want to start a training session?')


if __name__ =="__main__":
    main(sys.argv[1:])

