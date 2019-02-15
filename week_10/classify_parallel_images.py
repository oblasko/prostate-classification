from __future__ import print_function

import argparse

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Dense, GlobalMaxPooling2D, BatchNormalization, Dropout, ReLU, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import plot_model

from data import load_train_data, load_test_data, load_train_data_with_flnames

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.

def scale_images(imgs, mean, std):
    scaled_imgs = imgs.astype('float32')
    scaled_imgs -= mean
    scaled_imgs /= std

    return scaled_imgs

def create_parallel_imgs(imgs, masks, flnames):
    par_imgs_1 = []
    par_imgs_2 = []
    par_imgs_3 = []
    
    #testing purposes
    flnames_1 = []
    flnames_2 = []
    flnames_3 = []

    labels = []
    black_img = np.zeros(shape=(512,512,1))
    
    for id, flname_list in flnames.items():
        for i in range(len(flname_list)-1):
            if i == 0:
                sec_img = imgs[flname_list[i][0]]
                thrd_img = imgs[flname_list[i+1][0]]
                
                par_imgs_1.append(black_img)
                par_imgs_2.append(sec_img)
                par_imgs_3.append(thrd_img)

                flnames_1.append("black_img")
                flnames_2.append(flname_list[i][1])
                flnames_3.append(flname_list[i+1][1])
                    
                labels.append( masks[ flname_list[i][0] ].flatten().max() > 0)
    
            if i + 2 >= len(flname_list):
                first_img = imgs[flname_list[i][0]]
                sec_img = imgs[flname_list[i+1][0]]

                par_imgs_1.append(first_img)
                par_imgs_2.append(sec_img)
                par_imgs_3.append(black_img)
                
                flnames_1.append(flname_list[i][1])
                flnames_2.append(flname_list[i+1][1])
                flnames_3.append("black_img")
           
                labels.append( masks[ flname_list[i+1][0] ].flatten().max() > 0)
            else:    
                first_img = imgs[flname_list[i][0]]
                sec_img = imgs[flname_list[i+1][0]]
                thrd_img = imgs[flname_list[i+2][0]]

                par_imgs_1.append(first_img)
                par_imgs_2.append(sec_img)
                par_imgs_3.append(thrd_img)

                flnames_1.append(flname_list[i][1])
                flnames_2.append(flname_list[i+1][1])
                flnames_3.append(flname_list[i+2][1])
         
                
                labels.append( masks[ flname_list[i+1][0] ].flatten().max() > 0)
    
    return np.array(par_imgs_1), np.array(par_imgs_2), np.array(par_imgs_3), np.array(labels) #, flnames_1, flnames_2, flnames_3

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


def get_unet(dropout_rate):
    
    #parallel inputs for different images
    input1 = Input((img_rows, img_cols, 1))
    input2 = Input((img_rows, img_cols, 1))
    input3 = Input((img_rows, img_cols, 1))
    
    #first paraller layer
    conv1 = Conv2D(32, (3, 3), padding='same')(input1)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(32, (3, 3), padding='same')(conv1)
    conv1 = ReLU()(conv1)
    conv1 = MaxPooling2D((2, 2))(conv1)

    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
    conv1 = ReLU()(conv1)
    conv1 = MaxPooling2D((2, 2))(conv1)

    conv1 = Conv2D(128, (3, 3), padding='same')(conv1)
    conv1 = ReLU()(conv1)
    conv1 = Conv2D(128, (3, 3), padding='same')(conv1)
    conv1 = ReLU()(conv1)
    glb_pool1 = GlobalMaxPooling2D()(conv1)

    #second paraller layer
    conv2 = Conv2D(32, (3, 3), padding='same')(input2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(32, (3, 3), padding='same')(conv2)
    conv2 = ReLU()(conv2)
    conv2 = MaxPooling2D((2, 2))(conv2)

    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(64, (3, 3), padding='same')(conv2)
    conv2 = ReLU()(conv2)
    conv2 = MaxPooling2D((2, 2))(conv2)

    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = ReLU()(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = ReLU()(conv2)
    glb_pool2 = GlobalMaxPooling2D()(conv2)

    #third paraller layer
    conv3 = Conv2D(32, (3, 3), padding='same')(input3)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(32, (3, 3), padding='same')(conv3)
    conv3 = ReLU()(conv3)
    conv3 = MaxPooling2D((2, 2))(conv3)

    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(64, (3, 3), padding='same')(conv3)
    conv3 = ReLU()(conv3)
    conv3 = MaxPooling2D((2, 2))(conv3)

    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = ReLU()(conv3)
    conv3 = Conv2D(128, (3, 3), padding='same')(conv3)
    conv3 = ReLU()(conv3)
    glb_pool3 = GlobalMaxPooling2D()(conv3)

    #merge paraller layers
    mrg = concatenate([glb_pool1,glb_pool2,glb_pool3])

    if dropout_rate > 0.0:
        mrg = Dropout(dropout_rate)(mrg)

    # experiment with additional dense layer
    # try 256 or 128 nodes
    # relu activation function
    
    #output layer
    dense = Dense(1, activation='sigmoid')(mrg)

    model = Model(inputs=[input1,input2,input3], output=[dense])

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
    
    #scale masks to [0,1]
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    #create parallel images and generate labels
    train1, train2, train3, train_labels = create_parallel_imgs(imgs_train, imgs_mask_train, sorted_flnames_train)
    
  #   for i in range(len(f1)):
  #      print(f1[i] + "       " + f2[i] + "       " + f3[i] )
    
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization
    
    print("Train1 images: " + str(train1.shape))
    print("Train2 images: " + str(train2.shape))
    print("Train3 images: " + str(train3.shape))
    print("Train Labels : " + str(train_labels.shape))
    
    train1 = scale_images(train1, mean, std)
    train2 = scale_images(train2, mean, std)
    train3 = scale_images(train3, mean, std)
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet(args.dropout_rate)
    
    #plot the architecture of the model
    #plot_model(model, to_file='paraller_model.png')
    #return
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model.fit([train1,train2,train3],
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
    
    imgs_mask_test = imgs_mask_test.astype('float32')
    imgs_mask_test /= 255.  # scale masks to [0, 1]
  
    #concate neighbours and generate labels
    test1, test2, test3, test_labels = create_parallel_imgs(imgs_test, imgs_mask_test, sorted_flnames_test)
    
    scale_images(test1, mean, std)
    scale_images(test2, mean, std)
    scale_images(test3, mean, std)
    
    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    pred_labels = model.predict([test1, test2, test3],
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

    print("Test1 images: " + str(test1.shape))
    print("Test2 images: " + str(test2.shape))
    print("Test3 images: " + str(test3.shape))
    print("TestLabels : " + str(test_labels.shape))
    
    loss, accuracy = model.evaluate([test1, test2, test3],
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