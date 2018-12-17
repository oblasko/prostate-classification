from skimage.io import imread
import numpy as np
import os
from random import shuffle

#renaming masks after downloading the data

#for filename in os.listdir('../data/masks'):
#    if filename.endswith('.tif'):
#        os.rename('../data/masks/' + filename, '../data/masks/' + filename[5:-4] + '_mask' + '.tif')`      

def group_by_patient(path):
        images_by_patient = {}
        for file in os.listdir(path + "images"):
            if file.endswith(".tif"):
                split_file_name = file.split('_')
                patient_id = split_file_name[1]
            if patient_id in images_by_patient:
                images_by_patient[patient_id].append(file)
            else:
                images_by_patient[patient_id] = [file]
    
        for file in os.listdir(path + "masks"):
            if file.endswith(".tif"):
                split_file_name = file.split('_')
                patient_id = split_file_name[1]
            if patient_id in images_by_patient:
                images_by_patient[patient_id].append(file)
            else:
                images_by_patient[patient_id] = [file]
        return images_by_patient

def getImageId(filename):
    split_filename = filename.split('_')
    if len(split_filename) == 4:
        return split_filename[3][:-4]
    return split_filename[3]

def load_images(path):
    images = {}
    images_by_patient = group_by_patient(path)
    for key, img_list in images_by_patient.items():
        for i in range(0,len(img_list)):
            first_id = getImageId(img_list[i])
            for j in range(i,len(img_list)):
                sec_id = getImageId(img_list[j])
                if first_id == sec_id and img_list[i] != img_list[j]:
                     if key in images:
                        images[key].append((img_list[i], img_list[j]))
                     else:
                        images[key] = [(img_list[i], img_list[j])]
                     break
    return images

class Scan:
    def __init__(self, path, slice_file, mask_file, patient_id):
        self.patient_id = patient_id
        self.slice = imread(path + "images/" + slice_file, as_grey=True) 
        self.mask = imread(path + "masks/" + mask_file, as_grey=True)
        self.contains_prostate = self.get_label()
    
    def get_label(self):
        return self.mask.flatten().max() > 0

class Patient:
    def __init__(self, scans, patient_id):
        self.id = patient_id
        self.scans = scans
    def add_scan(self, scan):
        self.scans.append(scan)

class PatientsDB:
    def __init__(self, path):
        self.patients = self.load_patients(load_images(path), path)
    
    def load_patients(self, images, path):
        patients = []
        for patient_id, scans in images.items():
            P = Patient([], patient_id)
            patients.append(P)
            for slice_file, mask_file in scans:
                s = Scan(path, slice_file, mask_file, patient_id)
                P.add_scan(s)
        return patients
    
    def load_data(self):
        shuffle(self.patients)
        x_test = []
        x_train = []
        y_test = []
        y_train = []
        threshold = 0
        patients_len = len(self.patients)
        for i in range (0,patients_len):
            if threshold > (patients_len/100)*75:
                for scan in self.patients[i].scans:
                    x_test.append(scan.mask)
                    y_test.append(scan.contains_prostate)
            else:
                for scan in self.patients[i].scans:
                    x_train.append(scan.mask)
                    y_train.append(scan.contains_prostate)
            threshold+=1
        return ( ( np.array(x_train), np.array(y_train) ), (np.array(x_test), np.array(y_test)) )