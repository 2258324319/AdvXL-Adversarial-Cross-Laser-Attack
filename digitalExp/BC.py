import random
import time


from light_sim import img_simulate
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from multiprocessing import Pool


from bayes_opt import BayesianOptimization
from bayes_opt.event import DEFAULT_EVENTS, Events
from itertools import count
from bayes_opt import UtilityFunction
from get_obj_center import get_obj_center


utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

index = []
params = []

queries = 0

def predict_with_vgg19(model, img, cls_index):
    global queries
    def get_max(n, pre, pre_cla):
        predict = pre
        pre = np.argsort(-pre)
        top5_index = []
        for i in range(n):
            top5_index.append(int(pre[i]))
        return top5_index

    trans = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = trans(img).to('cuda:0')
    
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0) 
        predict = predict.to('cpu')
        predict_cla = torch.argmax(predict).numpy() 
        top5_index = get_max(1, predict, predict_cla)
        credit = predict[int(cls_index)]
    
    queries += 1

    return top5_index, credit

def predict_with_ResNet18(model, img, cls_index):
    global queries
    def get_max(n, pre, pre_cla):
        # 得到从大到小排序的索引号
        predict = pre
        pre = np.argsort(-pre)
        # 读取索引对应
        top5_index = []
        for i in range(n):
            top5_index.append(int(pre[i]))
        return top5_index

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = trans(img).to('cuda:0')
    # add dimension
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)  
        predict = predict.to('cpu')
        predict_cla = torch.argmax(predict).numpy()  
        top5_index = get_max(1, predict, predict_cla)
        credit = predict[int(cls_index)]
    
    queries += 1
    
    return top5_index, credit

def BC(img, cls_index, model, img_path, img_name, iter_nums):
    """
        iter_nums is a tuple as (centerPointsNumber, initNumber, fitNumber)
    """

    global flag,queries
    
    queries = 0
    
    
    pic_w = img.size[0]
    pic_h = img.size[1]
    width_top = min(pic_w, pic_h)/20



    # Bayes opt function
    def func_to_optimal(x, y, width, rotate):
        global index
        img_simulated_path = img_simulate(x, y, width, rotate, img_path)
        img = Image.open(img_simulated_path).convert("RGB")
    
        os.remove(img_simulated_path)
    
    #        _, res_vgg = predict_with_vgg19(model['VGG'],img, cls_index)
        index, res_resnet = predict_with_ResNet18(model, img, cls_index)
        print('res_resnet:'+str(float(res_resnet)), "correct?", index[0] == cls_index)
        return -float(res_resnet)

        
    attack_op = BayesianOptimization(
        func_to_optimal,
        {'x': (0, pic_w),
         'y': (0, pic_h),
         'width': (width_top/3, width_top),
         'rotate': (0, 45)}
    )

    time_serie = time.time()
    f = open('./opt_param.txt','w')
    f.close()
    # if not os.path.exists('./max_300/'+str(cls_index)):
    #     os.mkdir('./max_300/'+str(cls_index))

    cx ,cy = get_obj_center(img_path)
    width = random.uniform(width_top/4, width_top)
    rotate = random.uniform(0, 45)
    for i in range(iter_nums[0]):
        width = random.uniform(width_top/4, width_top)
        rotate = random.uniform(0, 45)
        attack_op.probe(
	        params={"x": cx+random.randint(-40,40), "y": cy+random.randint(-40,40), "width": width, "rotate": rotate},
            lazy=True,
	    )
    attack_op.maximize(n_iter=0, init_points=iter_nums[1]) #
    for i in range(iter_nums[2]):
        next_point = attack_op.suggest(utility)
        target = func_to_optimal(**next_point)
        params = next_point
        if cls_index != index[0]: 
            # credict_op = -opt_params_dict['target']
            x = params['x']
            y = params['y']
            width = params['width']
            rotate = params['rotate']
            print(next_point)
            print('Found a better light attack, credit:',-target)
            return x, y, width, rotate, queries#params returned
        else:
            attack_op.register(params=next_point, target=target)
    
    x = params['x']
    y = params['y']
    width = params['width']
    rotate = params['rotate']
    print('Fail to Found a better light attack')
    return x, y, width, rotate, queries#params returned
    
    
	



