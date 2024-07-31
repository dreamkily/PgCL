import argparse
import os
import torch
import torch.nn as nn
from SAM import SAM
from torch.optim.lr_scheduler import LambdaLR

def set_lambda(networks, lambda_):
    for n, l in zip(networks, lambda_):
        if n is None:
            continue
        n.set_lambda(l)
        

def get_optim_and_scheduler(main_model,dis_model, lr,lr_d, epochs, lr_steps, gamma):
        
    classifier_params=main_model.get_params(lr)
    dis_params=dis_model.get_params(lr_d)

    if not isinstance(lr_steps, list):
        lr_steps = [lr_steps,] 
    optimizer = torch.optim.SGD(dis_params, weight_decay=.0005, momentum=.9)
    clssifier_optimizer = SAM(classifier_params, base_optimizer=torch.optim.SGD,rho=0.01,lr=0.01,adaptive=False,weight_decay=.0005, momentum=.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_steps, gamma=gamma)

    
    return optimizer, scheduler,clssifier_optimizer

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                
def set_mode(model, mode="train"):
    if model is not None:
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eva()

def save_options(opt, save_folder):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    file_name = os.path.join(save_folder, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write(message)
        opt_file.write('\n')
