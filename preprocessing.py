"""
This Python class is specified for Pre-processing microscopy image datasets
- Preprocessing --> Class
- Source datasets are preprocessed using preprocess_Source_Data()
- Target dataset selections are selected and preprocessed using preprocess_Target_Data()
- To preprocess our 10 selections use reprocessFTandTestSamples()

"""

from collections import Counter
from WorkSpace import *
import numpy as np
import re
import ntpath
from PIL import  Image,ImageChops
import cv2
import matplotlib.pyplot as plt

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def get_slices(image, slice_size, steps_x,steps_y, remove_black_images, threshold):

    slices = []
    image_array = np.array(image)
    for x in range(0, image_array.shape[0], steps_x):
        for y in range(0, image_array.shape[1], steps_y):
            if x+slice_size <= image_array.shape[0] and y+slice_size <= image_array.shape[1]:
                img = image_array[x:x+slice_size, y:y+slice_size]
                if np.count_nonzero(img) < threshold and remove_black_images==True:
                    img = None
                slices.append(img)
    return slices


class Preprocessing():

    def __init__(self, selections,datasets, target_dir,
                 k_shots=None, crop=True,source_dir=os.getcwd()+'/Datasets/Raw/'):

        self.root_dir = source_dir
        self.target_dir = target_dir
        self.datasets = datasets
        self.selections = selections
        self.k_shots = k_shots
        self.crop = crop
        self.wrkspace = ManageWorkSpace(datasets=datasets)
        self.ft_crop_step = {'B5':{'x':30,'y':30},
                          'B39':{'x':50,'y':26},
                          'EM':{'x':30,'y':30},
                          'TNBC':{'x':28,'y':28},
                          'ssTEM':{'x':30,'y':30}
                          }


    def createFewShotTargetDirs(self):
        # Create Target Dataset Directories under
        # FewShotCellSegmentation/Dataset/FewShot/Target/

        for selection in self.selections:
            fewshot_target_dir = self.target_dir +'Target/'
            prefix = fewshot_target_dir+'Selection_' + str(selection) + '/'
            fine_tune_dir,test_dir = prefix + 'FinetuneSamples/',prefix + 'TestSamples/'


            for k_shot in self.k_shots:
                for dataset in self.datasets:
                    dirs = [fine_tune_dir+str(k_shot)+'-shot/'+dataset +'/Image/',
                            fine_tune_dir+str(k_shot)+'-shot/'+dataset +'/Groundtruth/',
                            test_dir+str(k_shot)+'-shot/'+dataset +'/Image/',
                            test_dir+str(k_shot)+'-shot/'+dataset +'/Groundtruth/']
                    preprocess_dirs = [fine_tune_dir+str(k_shot)+'-shot/'+'preprocessed/' + dataset+'/Image/',
                                       fine_tune_dir+str(k_shot)+'-shot/'+ 'preprocessed/' + dataset+'/Groundtruth/']
                    self.wrkspace.remove_dir(dirs+preprocess_dirs)
                    self.wrkspace.create_dir(dirs+preprocess_dirs)

    def createFewShotSourceDirs(self):
        # Create Source Dataset Directories under
        # FewShotCellSegmentation/Dataset/FewShot/Source/

        fewshot_source_dir = self.target_dir + 'Source/'
        for dataset in self.datasets:
            dirs = [fewshot_source_dir + dataset+ '/Image/',
                    fewshot_source_dir + dataset+'/Groundtruth/',
                    ]
            self.wrkspace.remove_dir(dirs)
            self.wrkspace.create_dir(dirs)


    def getRawImagesAndGroundtruth(self,dataset):

        image_files = []
        ground_truth_files = []

        ground_truth_prefix = '../Datasets/FewShot/Raw/' + dataset + '/Groundtruth/'
        image_prefix = '../Datasets/FewShot/Raw/' + dataset + '/Image/'

        if dataset == 'T5':
            ground_truth_folders = sorted(
                [folder + '/' for folder in os.listdir(ground_truth_prefix) if folder[0] == 'G'])
            image_folders = sorted(
                [folder + '/' for folder in os.listdir(image_prefix) if folder[0] == 'S'])

            for folder in ground_truth_folders:
                ground_truth_files += sorted([ground_truth_prefix + folder + f for f in
                                              os.listdir(ground_truth_prefix + folder) if
                                              f[0] != '.'])
            for folder in image_folders:
                image_files += sorted(
                    [image_prefix + folder + f for f in os.listdir(image_prefix + folder) if
                     f[0] != '.'])
        else:
            ground_truth_files = sorted(
                [ground_truth_prefix + f for f in os.listdir(ground_truth_prefix) if f[0] != '.'])
            image_files = sorted([image_prefix + f for f in os.listdir(image_prefix) if f[0] != '.'])[:len(ground_truth_files)]

        return image_files,ground_truth_files

    def getTestImagesandGroundtruth(self,dataset,fine_tune_dir):
        #Get Test Image samples which are not in Few-Shot samples

        image_files = []
        ground_truth_files = []
        ground_truth_prefix = '../Datasets/FewShot/Raw/' + dataset + '/Groundtruth/'
        image_prefix = '../Datasets/FewShot/Raw/' + dataset + '/Image/'

        if dataset == 'TNBC':
            ground_truth_folders = sorted(
                [folder + '/' for folder in os.listdir(ground_truth_prefix) if folder[0] == 'G'])
            image_folders = sorted(
                [folder + '/' for folder in os.listdir(image_prefix) if folder[0] == 'S'])

            for folder in image_folders:
                image_files += sorted([image_prefix + folder + f for f in os.listdir(image_prefix + folder)
                                       if f[0] != '.' and f not in os.listdir(fine_tune_dir+dataset+'/Image/')])

            for folder in ground_truth_folders:
                ground_truth_files += sorted([ground_truth_prefix + folder + f for f in
                                              os.listdir(ground_truth_prefix + folder) if
                                              f[0] != '.' and f not in os.listdir(fine_tune_dir+dataset+'/Groundtruth/')])
        else:
            ground_truth_files = sorted([ground_truth_prefix + f for f in os.listdir(ground_truth_prefix) if f[0] != '.' and
                 f not in os.listdir(fine_tune_dir+dataset+'/Groundtruth/')])

            image_files = sorted([image_prefix + f for f in os.listdir(image_prefix) if f[0] != '.' and
                 f not in os.listdir(fine_tune_dir+dataset+'/Image/')])[:
                len(ground_truth_files)]

        return image_files, ground_truth_files

    def savefiles(self,image_files,ground_truth_files,save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir + '/Image/')
            os.makedirs(save_dir + '/Groundtruth/')


        for file in image_files:
            image = Image.open(file)
            base_name = os.path.basename(file)
            image.save(save_dir + '/Image/' + base_name)

        for file in ground_truth_files:
            image = Image.open(file)
            base_name = os.path.basename(file)
            image.save(save_dir + '/Groundtruth/' + base_name)

    def selectKRandomShots(self,ground_truth_files,image_files,dataset,shot,
                           save_dir):
        # select K-shots from a dataset for fine-tuning
        isFileRepeated = True
        while isFileRepeated:
            ground_truth_temp = np.random.choice(ground_truth_files, shot)
            ground_truth_temp = sorted(ground_truth_temp)
            counter = Counter(ground_truth_temp)
            cnt_bol = []
            for key in counter:
                if counter[key] > 1:
                    cnt_bol.append(True)
            if len(cnt_bol) == 0:
                isFileRepeated = False
                ground_truth_files = ground_truth_temp

        ground_truth_files  = sorted(ground_truth_files)
        image_files = self.getKshotImageFiles(ground_truth_files,image_files,dataset)
        self.savefiles(image_files=image_files,ground_truth_files=ground_truth_files,save_dir=save_dir)

        return image_files,ground_truth_files

    def selectSortShots(self,ground_truth_files,image_files,dataset,
                           save_dir,idx):
        # select K-shots from a dataset for fine-tuning
        ground_truth_temp = []

        for shot_num in idx:
            ground_truth_temp.append(ground_truth_files[shot_num])#np.random.choice(ground_truth_files, shot)

        ground_truth_files = ground_truth_temp
        ground_truth_files  = sorted(ground_truth_files)
        image_files = self.getKshotImageFiles(ground_truth_files,image_files,dataset)
        self.savefiles(image_files=image_files,ground_truth_files=ground_truth_files,save_dir=save_dir)

        return image_files,ground_truth_files

    def selectUnlabelledPool(self, ground_truth_files, image_files, dataset,
                        save_dir, idx):
        # select K-shots from a dataset for fine-tuning
        ground_truth_temp = []
        for shot_num in range(len(ground_truth_files)):
            if shot_num not in idx:
                ground_truth_temp.append(ground_truth_files[shot_num])  # np.random.choice(ground_truth_files, shot)

        ground_truth_files = ground_truth_temp
        ground_truth_files = sorted(ground_truth_files)
        image_files = self.getKshotImageFiles(ground_truth_files, image_files, dataset)
        self.savefiles(image_files=image_files, ground_truth_files=ground_truth_files, save_dir=save_dir)

        return image_files, ground_truth_files
    def getKshotImageFiles(self,ground_truth_files,image_files,dataset):

        if dataset == 'ssTEM' or dataset == 'B39':
            fileCode = re.compile(r'\d+')
            base_name = []
            temp = []
            for ground_truth_file in ground_truth_files:
                base_name.append(path_leaf(ground_truth_file))
            for i in range(0, len(ground_truth_files)):
                for image_file in image_files:
                    if len(fileCode.findall(os.path.basename(image_file))) == 1:
                        if fileCode.findall(os.path.basename(image_file)) == fileCode.findall(base_name[i]):
                            temp.append(image_file)
                    elif fileCode.findall(os.path.basename(image_file)) == fileCode.findall(base_name[i]):
                        temp.append(image_file)
        else:
            base_name = []
            temp = []
            for ground_truth_file in ground_truth_files:
                base_name.append(os.path.basename(ground_truth_file))

            for image_file in image_files:
                if os.path.basename(image_file) in base_name:
                    temp.append(image_file)

        image_files = temp
        assert (len(ground_truth_files) == len(image_files))
        return image_files

    def preprocess_Groundtruth_pl(self,ground_truth_files,dataset, size=256, steps=None,
                               remove_black_images=False, threshold=360,crop=True,pseudo_label=True):

        ##TODO provide threshold of ssTEM and EM
        preprocessed_ground_truth = []
        preprocessed_ground_truth_pl = []
        for (file,image_file) in zip(ground_truth_files,self.image_files):
            ground_truth = Image.open(file)
            if dataset == 'ssTEM' and file.find('labels') != -1:
                ground_truth = ImageChops.invert(ground_truth)
                ground_truth = np.array(ground_truth)
                ground_truth[ground_truth >= 0.5] = 255
                ground_truth[ground_truth < 0.5] = 0
            elif dataset == 'B39':
                ground_truth = np.array(ground_truth)
                ground_truth = np.matmul(ground_truth, [0.2989, 0.5870, 0.1140, 0])
                ground_truth[ground_truth > 0] = 255
                ground_truth = ground_truth.astype(np.uint8)

                img = cv2.imread(image_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.equalizeHist(img)

                img[img > 0] = 255
                img[img < 0] = 0
                dilation = img
                ground_truth_pl = np.array(dilation)

            else:

                img = cv2.imread(image_file)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.equalizeHist(img)
                if dataset == 'TNBC':
                    _, thresh = cv2.threshold(img, 20, 110,
                                             cv2.THRESH_BINARY_INV) #TNBC:20,110 # 140,170,80 255
                elif dataset == 'ssTEM':
                    _, thresh = cv2.threshold(img, 140, 170,
                                          cv2.THRESH_BINARY_INV)  # TNBC:20,110 # 140,170,80 255
                elif dataset == 'EM':
                    _, thresh = cv2.threshold(img, 140, 170,
                                          cv2.THRESH_BINARY_INV)  # TNBC:20,110 # 140,170,80 255


                # Finding unknown region
                kernel = np.ones((2, 2), np.uint8)
                dilation = cv2.dilate(thresh,kernel,iterations=2)
                if dataset=='ssTEM':
                    dilation = thresh ##-->ssTEM
                dilation[dilation > 0] = 255
                dilation[dilation < 0] = 0

                ground_truth_pl = np.array(dilation)


                if dataset == 'B5':
                    ground_truth = np.array(ground_truth)
                    ground_truth_pl = np.zeros_like(ground_truth)
                    ground_truth[ground_truth >= 0.5] = 255
                    ground_truth[ground_truth < 0.5] = 0

            if crop:
                preprocessed_ground_truth += get_slices(ground_truth, size, steps_x=steps['x'], steps_y=steps['y'],
                                                        remove_black_images=remove_black_images,
                                                        threshold=threshold)
                preprocessed_ground_truth_pl += get_slices(ground_truth_pl, size, steps_x=steps['x'], steps_y=steps['y'],
                                                        remove_black_images=False,
                                                        threshold=threshold)

                for gt in range(0,len(preprocessed_ground_truth)):
                    if preprocessed_ground_truth[gt] is not None:
                        x = 0
                    else:
                        preprocessed_ground_truth_pl[gt] = None

            else:
                preprocessed_ground_truth += [ground_truth]
                preprocessed_ground_truth_pl += [ground_truth_pl]

        return preprocessed_ground_truth,preprocessed_ground_truth_pl

    def preprocess_Images(self,image_files,
                               size=256, steps=None, remove_black_images=False, threshold=360,crop=True):
        preprocessed_images = []
        if crop:
            for file in image_files:
                image = Image.open(file)
                image = image.convert('L')
                preprocessed_images += get_slices(image, size, steps_x=steps['x'], steps_y=steps['y'],
                                                  remove_black_images=remove_black_images, threshold=threshold)
        else:
            for file in image_files:
                image = Image.open(file)
                image = image.convert('L')
                image = np.array(image)
                preprocessed_images+=[image]

        return preprocessed_images

    def preprocess(self,ground_truth_files,image_files,dataset,
                   crop_window_size=256, steps=None,remove_black_images=False,crop=None,pseudo_label=True):
        self.image_files = image_files
        imgs = self.preprocess_Images(image_files, crop_window_size, steps=steps, remove_black_images=remove_black_images,
                               crop=crop)
        gt,gt_pl = self.preprocess_Groundtruth_pl(ground_truth_files, dataset, crop_window_size, steps=steps,
                                       remove_black_images=remove_black_images, crop=crop, pseudo_label=pseudo_label)
        return imgs,gt,gt_pl


    def save_preprocessed_data(self,preprocessed_ground_truth,preprocessed_ground_truth_pl,preprocessed_images,
                               save_dir,target=None):

        img_count = 0
        for i, sample in enumerate(preprocessed_ground_truth):
            if sample is not None:
                img_count += 1
                img = Image.fromarray(sample)
                img = img.convert('L')
                img.save(save_dir['groundtruth']+'ground_truth{}.png'.format(img_count))

        img_count = 0
        for i, sample in enumerate(preprocessed_ground_truth_pl):
            if sample is not None:
                img_count += 1
                img = Image.fromarray(sample)
                img = img.convert('L')
                img.save(save_dir['groundtruth_pl'] + 'ground_truth_pl{}.png'.format(img_count))

        img_count = 0
        for i, sample in enumerate(preprocessed_images):
            if preprocessed_ground_truth[i] is not None:
                img_count += 1
                img = Image.fromarray(sample)
                img.save(save_dir['image']+'image{}.png'.format(img_count))


    def preprocess_SortedTarget_Data(self,crop_window_size=256, crop_steps_dataset=None, remove_black_images=False,
                                experiment_name='test',shot=1,shot_pos=0,dataset=None,pseudo_label=False):

        # main function for extracting few-shot/Test samples selections and
        # pre-processing Target Data in the leave-one-dataset-out cross-validaton
        prefix = self.target_dir
        fine_tune_dir = prefix + 'FinetuneSamples/' + str(shot) + '-shot/'
        test_dir = prefix + 'TestSamples/' + str(shot) + '-shot/'

        save_ft_dir = {'image': fine_tune_dir + 'preprocessed/' + dataset + '/Image/',
                       'groundtruth': fine_tune_dir + 'preprocessed/' + dataset + '/Groundtruth/',
                       'groundtruth_pl': fine_tune_dir + 'preprocessed/' + dataset + '/Groundtruth_PL/'}

        save_test_dir = {'image': test_dir + dataset + '/Image/',
                         'groundtruth': test_dir + dataset + '/Groundtruth/',
                         'groundtruth_pl': test_dir + dataset + '/Groundtruth_PL/'}

        if os.path.exists(prefix):
            shutil.rmtree(prefix)

        os.makedirs(prefix+ 'crop_set/')
        for dir_ft,dir_test in zip(save_ft_dir,save_test_dir):
            os.makedirs(save_ft_dir[dir_ft])
            os.makedirs(save_test_dir[dir_test])


        image_files,ground_truth_files = self.getRawImagesAndGroundtruth(dataset)
        image_files,ground_truth_files = self.selectSortShots(ground_truth_files,image_files,dataset,idx=shot_pos,
                                                                 save_dir=fine_tune_dir+dataset)

        preprocessed_images,preprocessed_ground_truth,preprocessed_ground_truth_pl = self.preprocess(ground_truth_files,image_files,dataset,crop=True,
                                                                        crop_window_size=crop_window_size,steps=self.ft_crop_step[dataset],
                                                                        remove_black_images=remove_black_images,pseudo_label=pseudo_label)
        self.save_preprocessed_data(preprocessed_ground_truth=preprocessed_ground_truth,preprocessed_images=preprocessed_images,
                                    preprocessed_ground_truth_pl=preprocessed_ground_truth_pl,
                                    save_dir=save_ft_dir)

        test_images,test_groundtruth = self.getTestImagesandGroundtruth(dataset,fine_tune_dir)
        crop = False
        #save_dir = {'image':test_dir + dataset + '/Image/','groundtruth':test_dir + dataset + '/Groundtruth/','groundtruth_pl':test_dir + dataset + '/Groundtruth_PL/'}
        test_images, test_groundtruth,test_groundtruth_pl = self.preprocess(test_groundtruth,test_images,dataset,crop=crop,pseudo_label=pseudo_label)
        self.save_preprocessed_data(preprocessed_ground_truth=test_groundtruth,preprocessed_images=test_images,
                                    save_dir=save_test_dir,preprocessed_ground_truth_pl=test_groundtruth_pl)
        crop = True
        return fine_tune_dir,test_dir







