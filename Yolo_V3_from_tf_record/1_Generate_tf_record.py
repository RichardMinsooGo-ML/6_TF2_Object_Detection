import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from configuration import *

import glob
from datetime import datetime

import tqdm
import tensorflow as tf
from absl import app, flags, logging
# from absl.flags import FLAGS

from tools.voc_to_tfrecord import *

def main(argv):
    classes_map = {name: idx for idx, name in enumerate(
        open(classes).read().splitlines())}
    logging.info("Class mapping loaded: %s", classes_map)

    # tf_record_train = os.path.join(data_dir, 'MNIST_train.tfrecord')
    with tf.io.TFRecordWriter(tf_record_train) as writer:
        image_list = open(os.path.join(data_dir, 'train.txt')).read().splitlines()
        logging.info("Image list loaded: %d", len(image_list))

        counter = 0
        skipped = 0

        image_list[:] = [x for x in image_list if x]
        print(image_list[0:10])
        
        for image in tqdm.tqdm(image_list):
            
            image_file_train = os.path.join(image_dir_train, '%s.jpg' % image)
            annot_file_train = os.path.join(annot_dir_train, '%s.xml' % image)

            # processes the image and parse the annotation
            error, image_string = process_image(image_file_train)
            image_info_list = parse_annot(annot_file_train, classes_map)

            if not error:
                # convert voc to `tf.Example`
                example = create_tf_example(image_string, image_info_list)

                # write the `tf.example` message to the TFRecord files
                writer.write(example.SerializeToString())
                counter += 1
            else:
                skipped += 1

    print('{} : Wrote {} images to {}'.format( datetime.now(), counter, tf_record_train))

    with tf.io.TFRecordWriter(tf_record_val) as writer:
        image_list = open(os.path.join(data_dir, 'val.txt')).read().splitlines()
        logging.info("Image list loaded: %d", len(image_list))

        counter = 0
        skipped = 0

        image_list[:] = [x for x in image_list if x]
        print(image_list[0:10])

        for image in tqdm.tqdm(image_list):
            
            image_file_val = os.path.join(image_dir_val, '%s.jpg' % image)
            annot_file_val = os.path.join(annot_dir_val, '%s.xml' % image)

            # processes the image and parse the annotation
            error, image_string = process_image(image_file_val)
            image_info_list = parse_annot(annot_file_val, classes_map)

            if not error:
                # convert voc to `tf.Example`
                example = create_tf_example(image_string, image_info_list)

                # write the `tf.example` message to the TFRecord files
                writer.write(example.SerializeToString())
                counter += 1
            else:
                skipped += 1

    print('{} : Wrote {} images to {}'.format( datetime.now(), counter, tf_record_val))

if __name__ == '__main__':
    app.run(main)
