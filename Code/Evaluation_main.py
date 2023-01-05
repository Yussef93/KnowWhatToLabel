import sys
import os
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Code import Evaluation
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random

def addEvaluationArgs():
    parser = argparse.ArgumentParser(description="Evaluation Arguments")
    parser.add_argument("--finetune", type=int, default=1)
    parser.add_argument("--testpretrain", type=int, default=0)
    parser.add_argument("--testfinetune", type=int, default=1)
    parser.add_argument("--affine", type=int, default=0)
    parser.add_argument("--switchaffine", type=int, default=1)
    parser.add_argument("--plft", type=int, default=0)
    parser.add_argument("--targets",type=str,nargs="*",default=['TNBC'])
    parser.add_argument("--select", type=str, default='Ours',
                        help="Combination of B5,B39,TNBC,ssTEM,EM")
    parser.add_argument("--metamethods",type=str,nargs="*",default='BCE',
                        help="Combination of BCE,BCE_Entropy,BCE_Distillation,Combined")

    parser.add_argument("--metalr", type=float, default=1.0,
                        help="Pre-trained meta step size")

    parser.add_argument("--modellr", type=float, default=0.001,
                        help="Pre-trained learning rate")

    parser.add_argument("--finetune-lr", type=float, default=0.0001,
                        help="Finetune learning rate")
    parser.add_argument("--finetune-loss", type=str, default="weightedbce",
                        help="Loss function")
    parser.add_argument('--metaepochs', type=int, default=700)
    parser.add_argument('--innerepochs', type=int, default=30)
    parser.add_argument('--finetune-epochs', type=int, default=300)
    parser.add_argument('--statedictepoch', type=int, default=50)
    parser.add_argument('--num-shots', type=int,nargs="*",default=1)
    parser.add_argument("--pretrainedid", type=str, default='',
                        help="Additional identifier to Pretrained model's experiment name")

    parser.add_argument("--selectid", type=str,
                       default='test',
                       help="Additional identifier to selection experiment name")
    parser.add_argument("--finetuneid", type=str, default='',
                        help="Additional identifier to Finetuned model's experiment name")
    return parser

def evaluate_meta_learning(evaluation):
    evaluation.evaluate_meta_learning()

def main():

   parser = addEvaluationArgs()
   args = parser.parse_args()
   print(args)

   meta_params = {'methods': args.metamethods,
                  'hyperparams': {'meta_lr': str(args.metalr),
                                  'meta_epochs': str(args.metaepochs),
                                  'model_lr': str(args.modellr),
                                  'inner_epochs': str(args.innerepochs),
                                  'k-shot': '5',
                                  'optimizer': {'weight_decay': '0.0005',
                                                'momentum': '0.9'}}}

   batchsize_testset = {'TNBC': 32,
                        'B39': 32,
                        'ssTEM': 32,
                        'EM': 20,
                        'B5': 32}

   evaluation_config = {'targets': args.targets,
                        'k-shot': args.num_shots,
                        'batchsize_ftset': 64,
                        'batchsize_testset': batchsize_testset,
                        'ft_lr': args.finetune_lr,
                        'ft_epochs': args.finetune_epochs,
                        'optimizer': {'weight_decay': 0.0005,
                                      'momentum': 0.9, },
                        'Finetune': bool(args.finetune),
                        'Test_Pretrained':bool(args.testpretrain),
                        'Test_Finetuned':bool(args.testfinetune)}

   evaluation = Evaluation.Evaluation(evaluation_config=evaluation_config, meta_params=meta_params,
                                      args=args)

   evaluation.PL_sort_and_train()


if __name__ == '__main__':


    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()

