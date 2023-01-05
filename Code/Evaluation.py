

from torch.utils.data import DataLoader


from WorkSpace import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
import pickle
from Code import Models, Datasets
import logging
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from preprocessing import  Preprocessing
import gc
import os


def enable_dropout(model):
  for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()

class Evaluation():

    def __init__(self,args, evaluation_config,
                 meta_params=None):

        self.args = args
        self.wrkspace = ManageWorkSpace(datasets=evaluation_config['targets'])
        self.meta_params = meta_params
        self.evaluation_config = evaluation_config
        self.criterion= args.finetune_loss
        self.affine = args.affine
        self.switch_affine = args.switchaffine
        self.pretrained_experiment_name_postfix = args.pretrainedid
        self.ft_experiment_name_postfix = args.finetuneid

    def createFineTuneDirs(self):
        map_lr_method = self.wrkspace.map_dict['Meta_Learning']
        models_save_dir = '../models/' + map_lr_method + '/' + self.architecture + '/'
        logging_save_dir = '../Logging/' + map_lr_method + '/' + self.architecture + '/'
        self.wrkspace.create_dir([models_save_dir + 'Fine-tuned/',
                                  models_save_dir + 'Pre-trained/',
                                  logging_save_dir + 'Fine-tuned/',
                                  logging_save_dir + 'Pre-trained/'])

    def creatEvaluationMetaDirs(self):
        model_save_dir = '../models/Meta-models/' + self.architecture + '/'
        logging_save_dir = '../Logging/Meta-models/' + self.architecture + '/'

        for k_shot in self.evaluation_config['k-shot']:
            for meta_method in self.meta_params['methods']:
                for target in self.evaluation_config['targets']:
                    self.wrkspace.create_dir([model_save_dir + 'Fine-tuned/' + str(k_shot) + '-shot/' + meta_method +
                                              '/Target_' + target + '/',
                                              logging_save_dir + 'Fine-tuned/' + str(k_shot) + '-shot/' + meta_method +
                                              '/Target_' + target + '/FT_Loss_IoU/',
                                              logging_save_dir + 'Fine-tuned/' + str(k_shot) + '-shot/' + meta_method +
                                              '/Target_' + target + '/Test_Loss_IoU/',
                                              logging_save_dir + 'Pre-trained/' + str(k_shot) + '-shot/' + meta_method +
                                              '/Target_' + target + '/Test_Loss_IoU/'
                                              ])
        return model_save_dir, logging_save_dir

    def initModelECML(self):
            return Models.FCRNECML(in_channels=1, affine=self.affine,
                               sigmoid=True if self.criterion == 'bce' else False)


    def initModelECMLDropout(self):


        return Models.FCRNECML_dropout(in_channels=1, affine=self.affine,
                               sigmoid=True if self.criterion == 'bce' else False) \


    def load_model_state_dict(self, state_dict_path,epoch=None,target=None,method=None):

        model = self.initModelECML()

        if epoch != None:
            model.load_state_dict(torch.load(state_dict_path +'_' +str(epoch) +'_state_dict.pt',map_location=torch.device('cpu')))

        else:
            model.load_state_dict(torch.load(state_dict_path + '_state_dict.pt'))


        return model

    def getFTandTestLoader(self, selection_ft_path, selection_test_path, batchsize_ftset,
                           batchsize_testset, dataset):
        finetune_set = Datasets.CellsDataset(root_dir=selection_ft_path, dataset_selection=[dataset])
        testset = Datasets.CellsDataset(root_dir=selection_test_path, dataset_selection=[dataset])
        finetuneloader = DataLoader(finetune_set, batch_size=batchsize_ftset, shuffle=True)
        testloader = DataLoader(testset, batch_size=batchsize_testset[dataset],shuffle=False)

        return finetuneloader, testloader

    def getRawTargetDataLoader(self, data_path, dataset,batch_size=1):
        rawset = Datasets.CellsDataset(root_dir=data_path, dataset_selection=[dataset],transform=transforms.Compose([transforms.Grayscale(),
                                                                                                                     transforms.ToTensor(),
                                                                                                                     transforms.Normalize(mean=[0.5],std=[0.5])]))
        rawdataloader = DataLoader(rawset, batch_size=batch_size,shuffle=False)

        return rawdataloader


    def getExperimentName(self, k_shot='', target='', descr='finetuned',
                          selection='', lr_method=None):
        prefix = 'Meta_Learning' + '_' + descr + '_' + str(self.meta_params['hyperparams']['meta_lr']) + 'meta_lr_' + \
                 str(self.meta_params['hyperparams']['meta_epochs']) + 'meta_epochs_' + \
                 str(self.meta_params['hyperparams']['model_lr']) + 'model_lr_' + \
                 str(self.meta_params['hyperparams']['inner_epochs']) + 'inner_epochs_' + \
                 str(self.meta_params['hyperparams']['k-shot']) + 'shot' + self.ft_experiment_name_postfix


        experiment_name = prefix + str(k_shot) + 'shot_' + target + '_Selection_' + str(selection)

        return experiment_name, prefix

    def getPretrainedMetaModelName(self, meta_method, target):
        meta_pre_train_hyperparams = self.meta_params['hyperparams']
        prefix = 'Meta_Learning_' + meta_method + '_' + meta_pre_train_hyperparams['meta_lr'] + \
                 'meta_lr_' + meta_pre_train_hyperparams['model_lr'] + 'modellr_' + meta_pre_train_hyperparams[
                     'meta_epochs'] + \
                 'meta_epochs_' + meta_pre_train_hyperparams['inner_epochs'] + 'inner_epochs_' + \
                 meta_pre_train_hyperparams['k-shot'] + 'shot_'
        model_name = prefix + target + self.pretrained_experiment_name_postfix
        return model_name

    def getTargetandFTDir(self, selection_dir, k_shot):
        target_ft_dir = selection_dir + 'FinetuneSamples/' + str(k_shot) + '-shot/preprocessed/'
        target_test_dir = selection_dir + 'TestSamples/' + str(k_shot) + '-shot/'
        return target_ft_dir, target_test_dir

    def switchBatchnormParams(self,model):
        for m in model.modules():
            if isinstance(m, nn.Sequential):
                m[1] = nn.BatchNorm2d(m[1].num_features, affine=True)

        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                Models.init.constant_(m.weight, 0.1)
                Models.init.constant_(m.bias, 0)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                assert (torch.equal(m.weight.data, 0.1 * torch.ones_like(m.weight.data)))
                assert (torch.equal(m.bias.data, 0 * torch.ones_like(m.weight.data)))
        return model


    def init_model(self,target):
        model_save_dir = '../models/Meta-models/'
        meta_method = self.meta_params['methods']
        model_name = self.getPretrainedMetaModelName(meta_method, target)
        pre_trained_model_target_dir = model_save_dir + 'Pre-trained/' + meta_method + '/Target_'
        model_pre_train_state_dict_dir = pre_trained_model_target_dir + target + '/' + model_name + '/State_Dict/'
        model_pretrained = self.load_model_state_dict(state_dict_path=model_pre_train_state_dict_dir +
                                                                      model_name, epoch=300,
                                                      target=target, method=meta_method)

        if self.evaluation_config['Finetune'] and self.switch_affine:
            model_pretrained = self.switchBatchnormParams(model_pretrained)

        return model_pretrained
    def PL_sort_and_train(self):
        ## TODO: Add MC-dropout and entropy

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
        self.evaluation_config['ft_epochs'] = 20

        target = self.evaluation_config['targets'][0]
        patches = 0
        if target == 'TNBC':
            pos = np.array([i for i in range(0, 50)])
            patches = 100
        elif target == 'EM':
            pos = np.array([i for i in range(0, 165)])
            patches=400
        elif target =='ssTEM':
            pos = np.array([i for i in range(0, 20)])
            patches = 500
        elif target =='B5':
            pos = np.array([i for i in range(0, 1200)])
            patches = 100

        elif target =='B39':
            pos = np.array([i for i in range(0, 200)])
            patches = 100

        test_set_idx_path = './Test_sets/' + target + '/'
        if not os.path.exists(test_set_idx_path):
            os.makedirs(test_set_idx_path)

        ### if plft enabled then Pseudo-label cell seg will train
        if self.args.plft:
            pre_trained_model_target_dir = '../models/Pretrained/'
            model_name = target+'_model.pt'
            model_pretrained = self.load_model_state_dict(state_dict_path=pre_trained_model_target_dir +
                                                                          model_name, epoch=300)

            if self.evaluation_config['Finetune'] and self.switch_affine:
                model_pretrained = self.switchBatchnormParams(model_pretrained)


            model_scratch = self.initModelECML()
            model_scratch = self.switchBatchnormParams(model_scratch)
            model_scratch.load_state_dict(model_pretrained.state_dict())
            self.evaluation_config['ft_epochs'] = 100

            target_dir_pl = '../Datasets/FewShot/Sorted/' + target+'/PseudoLabel/' ## Dir to store PLs
            preprocessing_pl = Preprocessing(datasets=target, target_dir=target_dir_pl, selections=1)
            target_ft_dir, target_test_dir = preprocessing_pl.preprocess_SortedTarget_Data(
                experiment_name='Meta_Learning',
                shot=self.evaluation_config[
                    'k-shot'],
                dataset=target,
                remove_black_images=True,
                shot_pos=pos,
                pseudo_label=True)
            finetuneloader, _ = self.getFTandTestLoader(selection_ft_path=target_ft_dir + '/preprocessed/',
                                                        selection_test_path=target_test_dir,
                                                        batchsize_ftset=128,#self.evaluation_config['batchsize_ftset']
                                                        batchsize_testset=self.evaluation_config[
                                                            'batchsize_testset'],
                                                    dataset=target)
            _, _ = self.finetune(finetuneloader=finetuneloader, model=model_scratch,
                                 save_path='', logger=logger)

            torch.save(model_scratch.state_dict(), '../models/PLFT/model_pretrained_MetaBCE_100epochs_PL_target_' + target + '.pth')

            del model_scratch
            gc.collect()
        self.evaluation_config['ft_epochs'] = 20

        iou_ours = []
        iou_random = []
        iou_test = []


        rawset = Datasets.CellsDataset(root_dir='../Datasets/FewShot/Raw/', dataset_selection=[target],
                                       transform=transforms.Compose([transforms.Grayscale(),
                                                                     transforms.ToTensor(),
                                                                     transforms.Normalize(mean=[0.5], std=[0.5])]))

        for set in range(0, 10):
            if isinstance(self.evaluation_config['k-shot'], list):
                self.evaluation_config['k-shot'] = self.evaluation_config['k-shot'][0]

            target_dir_unlabelled = '../Datasets/FewShot/Sorted/'+target+ self.args.selectid + str(
                    self.evaluation_config['k-shot']) + '_shots/'


            if not os.path.isfile(test_set_idx_path + '20percent_testset_' + str(set) +'.pth'):
                p_test = list(np.random.choice(pos, size=int(0.2 * len(pos)), replace=False))
                p_unlabelled = [p for p in pos if p not in p_test]
                torch.save(p_test, test_set_idx_path + '20percent_testset_' + str(set) + '.pth')
                torch.save(p_unlabelled, test_set_idx_path + 'unlabelledset_' + str(set) + '.pth')

            else:
                p_test = torch.load(test_set_idx_path + '20percent_testset_' + str(set) + '.pth')
                p_unlabelled = torch.load(test_set_idx_path + 'unlabelledset_' + str(set) + '.pth')

            if isinstance(self.evaluation_config['k-shot'], list):
                self.evaluation_config['k-shot'] = self.evaluation_config['k-shot'][0]

            if not os.path.isfile(test_set_idx_path+'randomset_'+str(set)+'_'+str(self.evaluation_config['k-shot'])+
                                  '_shot.pth'):
                p_random = list(np.random.choice(p_unlabelled, size=self.evaluation_config['k-shot'], replace=False))

                torch.save(p_random, test_set_idx_path + 'randomset_' + str(set) + '_' + str(self.evaluation_config['k-shot']) +
                           '_shot.pth')
            elif self.evaluation_config['k-shot'] == 5 and target =='TNBC':
                p_random = torch.load(test_set_idx_path + 'randomset_' + str(set) + '.pth')

            else:

                p_random = torch.load(
                    test_set_idx_path + 'randomset_' + str(set) + '_' + str(self.evaluation_config['k-shot']) +
                    '_shot.pth')

            assert (not np.array_equal(np.array(p_test), np.array(p_random)))
            assert (not np.array_equal(np.array(p_test), np.array(p_unlabelled)))
            testloader = torch.utils.data.DataLoader(rawset,
                                                     batch_size=len(p_test) if target !='B5' else 32,
                                                     # self.evaluation_config['batchsize_testset'][target],
                                                     sampler=torch.utils.data.SubsetRandomSampler(p_test))
            if self.args.select == 'test':
                model_scratch = self.initModelECML()
                model_scratch = self.switchBatchnormParams(model_scratch)

                model_scratch.load_state_dict(
                    torch.load('../models/PLFT/model_pretrained_MetaBCE_100epochs_PL_target_' + target + '.pth'), strict=True)
                _, iou, _ = self.test(model_scratch, testloader, num_test=len(p_test))
                gc.collect()
                torch.cuda.empty_cache()
                iou_test.append(iou)


            if self.args.select == 'Random':

                target_dir_random = '../Datasets/FewShot/Sorted/'+target+'/CropRandomSet_ModelPLFT/' + str(
                    self.evaluation_config['k-shot']) + '_shots/'+'Randomset_' +str(set)+'/'

                if not os.path.isfile(target_dir_random+'crop_set/crop_set'+str(set)+'.pth'):
                    preprocessing_random = Preprocessing(datasets=target, target_dir=target_dir_random, selections=1)
                    target_ft_dir, target_test_dir = preprocessing_random.preprocess_SortedTarget_Data(
                        shot=self.evaluation_config[
                            'k-shot'],
                        dataset=target,
                        remove_black_images=True,
                        shot_pos=p_unlabelled,
                        pseudo_label=True)

                    finetuneloader, _ = self.getFTandTestLoader(selection_ft_path=target_ft_dir + '/preprocessed/',
                                                                selection_test_path=target_test_dir,
                                                                batchsize_ftset=self.evaluation_config[
                                                                    'batchsize_ftset'],
                                                                batchsize_testset=self.evaluation_config[
                                                                    'batchsize_testset'],
                                                                dataset=target)

                    random_pos = np.random.randint(low=0,high=len(finetuneloader.dataset)-1,size=patches*self.evaluation_config['k-shot'])
                    random_pos = list(random_pos)
                    finetune_set = Datasets.CellsDataset(root_dir=target_ft_dir + '/preprocessed/', dataset_selection=[target])
                    finetuneloader = torch.utils.data.DataLoader(finetune_set,batch_size=1,sampler=torch.utils.data.SubsetRandomSampler(random_pos))
                    images, gt, gt_pl = torch.zeros((len(random_pos), 1, 256, 256)), torch.zeros(
                        (len(random_pos), 1, 256, 256)), torch.zeros(
                        (len(random_pos), 1, 256, 256))

                    for idx, batch in enumerate(finetuneloader, 0):
                        with torch.no_grad():
                            images[idx] = batch[0]
                            gt[idx] = batch[1]
                            gt_pl[idx] = batch[2]
                    crop_set = torch.utils.data.TensorDataset(images, gt, gt_pl)
                    del images,gt_pl,gt
                    gc.collect()

                    assert (len(crop_set) == patches * self.evaluation_config['k-shot'])
                    os.makedirs(target_dir_random+'crop_set/')
                    torch.save(crop_set,target_dir_random+'crop_set/crop_set'+str(set)+'.pth')
                else:
                    crop_set = torch.load(target_dir_random+'crop_set/crop_set'+str(set)+'.pth')

                    images, gt, gt_pl = torch.zeros((0, 1, 256, 256)), torch.zeros((0, 1, 256, 256)), \
                                    torch.zeros( (0, 1, 256, 256))
                    for k in range(0,patches*self.evaluation_config['k-shot']):
                        images = torch.cat([images, crop_set[k][0].unsqueeze(dim=0)], dim=0)
                        gt = torch.cat([gt, crop_set[k][1].unsqueeze(dim=0)], dim=0)
                        gt_pl = torch.cat([gt_pl, crop_set[k][2].unsqueeze(dim=0)], dim=0)
                    crop_set = torch.utils.data.TensorDataset(images, gt, gt_pl)
                    assert(len(crop_set)==patches*self.evaluation_config['k-shot'])

                finetuneloader = torch.utils.data.DataLoader(crop_set, batch_size=self.evaluation_config[
                    'batchsize_ftset'], shuffle=True)

                print("---Fine-Tuning Pretrained Model (Randomset) Set {}---".format(set+1))
                model_scratch = self.initModelECML()
                model_scratch = self.switchBatchnormParams(model_scratch)

                model_scratch.load_state_dict( torch.load('../models/PLFT/model_pretrained_MetaBCE_100epochs_PL_target_' + target + '.pth'),strict=True)

                _, _ = self.finetune(finetuneloader=finetuneloader,
                                    model=model_scratch,
                                     save_path='', logger=logger)
                _, iou, _ = self.test(model_scratch, testloader,num_test=len(p_test))
                torch.cuda.empty_cache()
                gc.collect()
                iou_random.append(iou)
                if os.path.exists(target_dir_random + 'FinetuneSamples/'):
                    shutil.rmtree(target_dir_random + 'FinetuneSamples/')
                    shutil.rmtree(target_dir_random + 'TestSamples/')


            if self.args.select == 'Ours':
                preprocessing_ours = Preprocessing(datasets=target, target_dir=target_dir_unlabelled, selections=1)
                target_ft_dir, target_test_dir = preprocessing_ours.preprocess_SortedTarget_Data(
                    shot=self.evaluation_config[
                        'k-shot'],
                    dataset=target,
                    remove_black_images=True,
                    shot_pos=p_unlabelled,
                    pseudo_label=True)
                if not os.path.exists(target_dir_unlabelled + 'crop_set/'):
                    os.makedirs(target_dir_unlabelled + 'crop_set/')
                if not os.path.isfile(target_dir_unlabelled + 'crop_set/crop_labelledset' + str(set) + '.pth'):

                    finetuneloader, _ = self.getFTandTestLoader(selection_ft_path=target_ft_dir + '/preprocessed/',
                                                                selection_test_path=target_test_dir,
                                                                batchsize_ftset=256,
                                                                batchsize_testset=self.evaluation_config[
                                                                    'batchsize_testset'],
                                                                dataset=target)

                    model_scratch = self.initModelECML()
                    model_scratch = self.switchBatchnormParams(model_scratch)

                    if not os.path.isfile('./model_scratch_sameInit.pth'):

                        torch.save(model_scratch.state_dict(), '../models/model_scratch_sameInit.pth')
                    else:
                        model_scratch.load_state_dict(torch.load('../models/model_scratch_sameInit.pth'))

                    model_scratch.load_state_dict(torch.load('../models/PLFT/model_pretrained_MetaBCE_100epochs_PL_target_' + target + '.pth'))

                    confidence_cnt = torch.zeros((len(finetuneloader.dataset), 1))

                    model_scratch.cuda()
                    model_scratch.eval()
                    for idx, batch in enumerate(finetuneloader, 0):

                        with torch.no_grad():
                            out = {}
                            for augment in batch[3].keys():
                                batch[3][augment]=batch[3][augment].cuda()
                                out[augment],_ = model_scratch(batch[3][augment])

                            pgt = torch.nn.Sigmoid()(out['original'])
                            pgt[pgt >= 0.5] = 1
                            pgt[pgt < 0.5] = 0
                            loss = nn.BCEWithLogitsLoss(reduction='none')(out['contrast'], pgt)
                            loss += nn.BCEWithLogitsLoss(reduction='none')(out['brightness'], pgt)
                            loss += nn.BCEWithLogitsLoss(reduction='none')(out['sharpness'], pgt)

                            confidence_cnt[batch[4]] = (loss.sum(dim=[1,2,3])/(256*256)).unsqueeze(dim=1).cpu()


                    model_scratch.to('cpu')
                    torch.cuda.empty_cache()
                    gc.collect()

                    _, top_patches = torch.topk(confidence_cnt, dim=0,
                                                k=patches * self.evaluation_config['k-shot'],
                                                largest=True)
                    top_patches = [p.item() for p in top_patches[:patches * self.evaluation_config['k-shot']]]
                    finetune_set = Datasets.CellsDataset(root_dir=target_ft_dir + '/preprocessed/',dataset_selection=[target])
                    finetuneloader = torch.utils.data.DataLoader(finetune_set, batch_size=1,
                                                                sampler=torch.utils.data.SubsetRandomSampler(
                                                                     top_patches))
                    images_top, gt_top, gt_pl_top = torch.zeros((len(top_patches), 1, 256, 256)), torch.zeros(
                    (len(top_patches), 1, 256, 256)), torch.zeros(
                    (len(top_patches), 1, 256, 256))
                    for idx, batch in enumerate(finetuneloader, 0):
                        with torch.no_grad():
                            images_top[idx] = batch[0]
                            gt_top[idx] = batch[1]
                            gt_pl_top[idx] = batch[2]

                    crop_labelledset = torch.utils.data.TensorDataset(images_top, gt_top, gt_top)

                else:
                    top_patches = torch.load(target_dir_unlabelled + 'crop_set/top_patches' + str(set) + '.pth')
                    top_patches = [p.item() for p in top_patches[:patches * self.evaluation_config['k-shot']]]
                    finetune_set = Datasets.CellsDataset(root_dir=target_ft_dir + '/preprocessed/',
                                                         dataset_selection=[target])
                    finetuneloader = torch.utils.data.DataLoader(finetune_set, batch_size=1,
                                                                 sampler=torch.utils.data.SubsetRandomSampler(
                                                                     top_patches))
                    images_top, gt_top, gt_pl_top = torch.zeros((len(top_patches), 1, 256, 256)), torch.zeros(
                        (len(top_patches), 1, 256, 256)), torch.zeros(
                        (len(top_patches), 1, 256, 256))
                    for idx, batch in enumerate(finetuneloader, 0):
                        with torch.no_grad():
                            images_top[idx] = batch[0]
                            gt_top[idx] = batch[1]
                            gt_pl_top[idx] = batch[2]

                    crop_labelledset = torch.utils.data.TensorDataset(images_top, gt_top, gt_top)
                finetuneloader = torch.utils.data.DataLoader(crop_labelledset, batch_size=self.evaluation_config[
                    'batchsize_ftset'], shuffle=True)


                print("---Fine-Tuning Pretrained Model (Ours) Set {}---".format(set+1))
                model_scratch = self.initModelECML()
                model_scratch = self.switchBatchnormParams(model_scratch)
                model_scratch.load_state_dict( torch.load('../models/PLFT/model_pretrained_MetaBCE_100epochs_PL_target_' + target + '.pth'), strict=True)

                _, _ = self.finetune(finetuneloader=finetuneloader,
                                    model=model_scratch,
                                     save_path='', logger=logger)

                _, iou, _ = self.test(model_scratch, testloader,num_test=len(p_test))
                model_scratch.to('cpu')
                torch.cuda.empty_cache()
                gc.collect()
                iou_ours.append(iou)


        results_path_random = '../Datasets/FewShot/Sorted/' + target + self.args.selectid + str(
            self.evaluation_config['k-shot']) + '_shots/Randomresults/'
        results_path_ours = '../Datasets/FewShot/Sorted/'+target+ self.args.selectid+ str(
                    self.evaluation_config['k-shot']) + '_shots/results/'
        if not os.path.exists(results_path_random):
            os.makedirs(results_path_random)

        if not os.path.exists(results_path_ours):
            os.makedirs(results_path_ours)


        if self.args.selct == 'Random':

            print("Set:{} Dataset:{} NumPatches:{} Random IoU: {} +/- {}".format(set+1,target,len(crop_set),np.array(iou_random).mean()*100,np.array(iou_random).std()*100))

            torch.save(iou_random, results_path_random + 'iou_random_MetaBCE_PLFTModel_20percentest_'+str(set+1)+'.pth')

    def calc_weights(self,labels):
        pos_tensor = torch.ones_like(labels)

        for label_idx in range(0,labels.size(0)):
            pos_weight = torch.sum(labels[label_idx]==1)
            neg_weight = torch.sum(labels[label_idx]==0)
            #pos_weight = torch.sum(labels[label_idx]==torch.max(labels).item())
            #neg_weight = torch.sum(labels[label_idx]==torch.min(labels).item())
            ratio = float(neg_weight.item()/pos_weight.item())
            pos_tensor[label_idx] = ratio*pos_tensor[label_idx]

        return pos_tensor

    def finetune(self, finetuneloader, model, save_path, logger=None,writer=None):
        finetune_loss = 0
        iou_finetune = 0
        acc_finetune = 0
        total_foreground = 0
        finetune_loss_epoch = []
        finetune_iou_epoch = []
        num_samples = 0
        ft_epochs = self.evaluation_config['ft_epochs']

        optimizer = optim.Adam(model.parameters(), lr=self.evaluation_config['ft_lr'],
                               weight_decay=self.evaluation_config['optimizer']['weight_decay'])
        model.cuda()

        for e in range(ft_epochs):
            model.train()
            for images, labels, _ in finetuneloader:
                images, labels = images.cuda(), labels.cuda()
                optimizer.zero_grad()

                output, _ = model(images)

                iou_temp, intersection_temp, union_temp, acc_temp = self.intersection_over_union(output, labels)
                if self.criterion == 'bce':
                   loss = nn.BCELoss()(output, labels)
                else:
                   loss = nn.BCEWithLogitsLoss(pos_weight=self.calc_weights(labels))(output,
                                                                                     labels)

                loss.backward()
                optimizer.step()

                finetune_loss += loss.item() * images.size(0)
                iou_finetune += iou_temp.item() * images.size(0)
                acc_finetune += torch.sum(acc_temp).item()
                total_foreground += torch.sum(labels == 1).item()
                num_samples += images.size(0)


            finetune_loss = finetune_loss / len(finetuneloader.dataset)
            iou_finetune = iou_finetune / len(finetuneloader.dataset)
            acc_finetune = acc_finetune / total_foreground
            logger.info('Epoch:{}//{} \tTrain loss: {:.4f}\tTrain IOU: {:.4f}\t FCA: {:.4f}'.format(e + 1, ft_epochs,
                                                                                                    finetune_loss,iou_finetune,
                                                                                               acc_finetune))

            finetune_loss_epoch.append(finetune_loss)
            finetune_iou_epoch.append(iou_finetune)

        return finetune_loss_epoch, finetune_iou_epoch

    def test(self, model, testloader,num_test):
        iou = 0
        acc = 0
        total_foreground = 0
        test_loss = 0
        model.eval()
        model.cuda()
        for child in model.children():
            if type(child) == nn.Sequential:
                for ii in range(len(child)):
                    if type(child[ii]) == nn.BatchNorm2d:
                        child[ii].track_running_stats = False

        test_start = time.time()
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                output, _ = model(images)
            output=nn.Sigmoid()(output)
            try:
                loss = nn.BCELoss()(output, labels)

            except Exception:
                loss = nn.BCELoss()(output, labels.float())
            iou_temp, intersection_temp, union_temp, acc_temp = self.intersection_over_union(output, labels)

            test_loss += loss.item() * images.size(0)
            iou += iou_temp.item()* images.size(0)

            acc += torch.sum(acc_temp).item()
            total_foreground += torch.sum(labels == 1).item()
        test_end = time.time()
        test_loss = test_loss / num_test
        iou = iou / num_test

        acc = acc / total_foreground
        print('Test Loss: {:.4f} \tTest IOU: {:.4f}\tFCA: {:.4f}\tTest Time: {:.3f} min\n'.format(test_loss, iou, acc,
                                                                                                  (test_end - test_start) / 60))

        return test_loss, iou, acc


    def intersection_over_union(self, tensor, labels, device=torch.device("cuda:0")):
        iou = 0
        foreground_acc = 0

        labels_tens = labels.type(torch.BoolTensor)
        ones_tens = torch.ones_like(tensor, device=device)
        zeros_tens = torch.zeros_like(tensor, device=device)
        if tensor.shape[0] > 1:
            temp_tens = torch.where(tensor >= 0.5, ones_tens, zeros_tens)
            intersection_tens = (temp_tens.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum((1, 2))

            union_tens = (temp_tens.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum((1, 2))
            iou += torch.mean((intersection_tens + 0.0001) / (union_tens + 0.0001))
            foreground_acc += intersection_tens
        else:
            temp_tens = torch.where(tensor >= 0.5, ones_tens, zeros_tens)
            intersection_tens = (temp_tens.squeeze().type(torch.BoolTensor) & labels_tens.squeeze()).float().sum()
            union_tens = (temp_tens.squeeze().type(torch.BoolTensor) | labels_tens.squeeze()).float().sum()
            iou += torch.sum((intersection_tens + 0.0001) / (union_tens + 0.0001))
            foreground_acc += intersection_tens

        del temp_tens
        del labels_tens
        del ones_tens
        del zeros_tens
        torch.cuda.empty_cache()
        total_iou = iou
        return total_iou,torch.sum(intersection_tens).item(),torch.sum(union_tens).item(), foreground_acc

    def save_result(self, logging_dir, result, experiment_name):
        f_pickle = open(logging_dir + experiment_name + '.pickle', 'wb')
        f_csv = open(logging_dir + experiment_name + '.csv', 'w')
        pickle.dump(result, f_pickle)
        df = pd.DataFrame(result)
        df.to_csv(f_csv, header=False, index=False)
        f_pickle.close()
        f_csv.close()


