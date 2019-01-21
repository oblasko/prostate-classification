from __future__ import print_function

import argparse

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D, BatchNormalization, Dropout, ReLU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

from data import load_train_data, load_test_data, load_train_data_with_flnames

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

#extracts the patient id from the filename of the image
def getPatientId(filename):
    split_filename = filename.split('_')
    return split_filename[1]

#extracts the slice id from the filename of the image
def getSliceId(filename):
    split_filename = filename.split('_')
    return split_filename[3][:-4]

#groups all corresponding image indexes by patient
def split_images_by_patient(imgs, masks, flnames):
    indexes_by_patient = {}
    for i in range(len(imgs)):
        patient_id = getPatientId(flnames[i])
        if patient_id in indexes_by_patient:
            indexes_by_patient[patient_id].append((i, flnames[i]))
        else:
            indexes_by_patient[patient_id] = [(i, flnames[i])]
    return indexes_by_patient

#sorting criteria
def get_key(item):
    return int(getSliceId(item[1]))

#sorts the filenames and indexes by slice id
def sort_by_slice(dict):
    for id, flnames in dict.items():
        dict[id] = sorted(flnames, key=get_key)
    return dict

#concates adjacent images in the following order: 1-2-3, 2-3-4, 3-4-5, 4-5-6, ...
def create_concated_imgs(imgs, flnames):
    concated_imgs = []
    black_img = np.zeros(shape=(512,512,1))
    for id, flname_list in flnames.items():
        for i in range(len(flname_list)):
            if i + 1 >= len(flname_list):
                first_img = imgs[flname_list[i][0]]
                concated_imgs.append(np.concatenate( (first_img, black_img, black_img), axis=2))
            elif i + 2 >= len(flname_list):
                first_img = imgs[flname_list[i][0]]
                sec_img = imgs[flname_list[i+1][0]]
                concated_imgs.append(np.concatenate( (first_img, sec_img, black_img), axis=2))
            else:    
                first_img = imgs[flname_list[i][0]]
                sec_img = imgs[flname_list[i+1][0]]
                thrd_img = imgs[flname_list[i+2][0]]
                concated_imgs.append(np.concatenate( (first_img, sec_img, thrd_img), axis=2))
    return np.array(concated_imgs)      

#generates label from masks
def generate_labels(masks):
    labels = []
    for i in range(1, len(masks)):
        #starting at index 1 because we always generate the label from the middle mask
        labels.append( masks[i].flatten().max() > 0 )
    
    #no neigbour at the last image, mask is black image here -> no prostate
    labels.append(False)   
    return np.array(labels)

def get_unet(dropout_rate):
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = ReLU()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = ReLU()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = ReLU()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = ReLU()(conv4)
    conv4 = Conv2D(256, (3, 3), padding='same')(conv4)
    conv4 = ReLU()(conv4)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv5 = ReLU()(conv5)
    conv5 = Conv2D(512, (3, 3), padding='same')(conv5)
    conv5 = ReLU()(conv5)

    pooling = GlobalMaxPooling2D()(conv5)

    if dropout_rate > 0.0:
        pooling = Dropout(dropout_rate)(pooling)

    dense1 = Dense(1, activation='sigmoid')(pooling)

    model = Model(inputs=[inputs], outputs=[dense1])

    loss = "binary_crossentropy"
    model.compile(optimizer=Adam(lr=1e-5),
                  loss=loss,
                  metrics=["accuracy"])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]

    return imgs_p


def train_and_predict(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pred_dir = os.path.join(args.output_dir,
                            "preds")
        
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    
    #load images and masks as numpy arrays (N_images, 512, 512) and filenames as numpy array (N_images)
    imgs_train, imgs_mask_train, imgs_flnames_train = load_train_data_with_flnames(args.input_dir)

    #change shape of images to (N_images, 512, 512, 1)
    imgs_train = preprocess(imgs_train)
    
    #change shape of masks to (N_images, 512, 512, 1)
    imgs_mask_train = preprocess(imgs_mask_train)
    
    #get dictionary of patient ids and corresponding indexes of images and masks
    images_by_patient_train = split_images_by_patient(imgs_train, imgs_mask_train, imgs_flnames_train)
    
    #sort the splitted images by slice id
    sorted_flnames_train = sort_by_slice(images_by_patient_train)
    
    #concate neighbours
    imgs_train = create_concated_imgs(imgs_train, sorted_flnames_train)
    
    print("Train shape")
    print(imgs_train.shape)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    train_labels = generate_labels(imgs_mask_train)
   
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet(args.dropout_rate)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit(imgs_train,
              train_labels,
              batch_size=args.batch_size,
              epochs=args.num_epochs,
              verbose=1,
              shuffle=True,
              validation_split=args.validation_split)

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    
    #load images and masks as numpy arrays (N_images, 512, 512) and filenames as numpy array (N_images)
    imgs_test, imgs_mask_test, imgs_flname_test = load_test_data(args.input_dir)

    #change shape of images and masks to (N_images, 512, 512, 1)
    imgs_test = preprocess(imgs_test)
    imgs_mask_test = preprocess(imgs_mask_test)
    
    #get dictionary of patient ids and corresponding indexes of images and masks
    images_by_patient_test = split_images_by_patient(imgs_test, imgs_mask_test, imgs_flname_test)
    
    #sort the splitted images by slice id
    sorted_flnames_test= sort_by_slice(images_by_patient_test)
    
    #concate neighbours
    imgs_test = create_concated_imgs(imgs_test, sorted_flnames_test)
    
    print("Test shape")
    print(imgs_test.shape)


    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]
    test_labels = generate_labels(imgs_mask_test)
    
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    pred_labels = model.predict(imgs_test,
                                verbose=1,
                                batch_size = args.batch_size)

    # Predictions need to be thresholded
    binary = np.zeros(pred_labels.shape)
    binary[pred_labels > 0.5] = 1.0

    np.save(os.path.join(args.output_dir, 'pred_image_classes.npy'),
            binary)

    print('-'*30)
    print('Evaluating model on test data...')
    print('-'*30)
    loss, accuracy = model.evaluate(imgs_test,
                                    test_labels,
                                    batch_size = args.batch_size)

    print("Loss:", loss)
    print("Accuracy:", accuracy)

def parseargs():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-epochs",
                        type=int,
                        required=True)

    parser.add_argument("--batch-size",
                        type=int,
                        default=8)

    parser.add_argument("--dropout-rate",
                        type=float,
                        default=0.75)
    
    parser.add_argument("--validation-split",
                        type=float,
                        default=0.2)

    parser.add_argument("--input-dir",
                        type=str,
                        required=True)

    parser.add_argument("--output-dir",
                        type=str,
                        required=True)

    return parser.parse_args()
    
if __name__ == '__main__':
    args = parseargs()
    train_and_predict(args)
