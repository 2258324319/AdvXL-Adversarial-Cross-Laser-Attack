from light_sim import img_simulate
from VGG19.predict import predict_with_vgg19
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from BC import BC
from multiprocessing import Pool
import multiprocessing
import shutil
import sys



def predict(model, img):
    def get_max(n, pre, pre_cla):
        # 得到从大到小排序的索引号
        predict = pre
        pre = np.argsort(-pre)
        # 读取索引对应
        with open('./VGG19/imagenet1000_clsidx_to_labels.txt', 'r') as f:
            line = f.readlines()
        top1_index = []
        for i in range(n):
            top1_index.append(int(pre[i]))

        return top1_index

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = trans(img)
    img = img.to('cuda:0')
    # 加一个batch维度
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)  # 得到概率分布
        predict = predict.to('cpu')
        predict_cla = torch.argmax(predict)
        predict_cla = predict_cla.to('cpu')
        predict_cla = predict_cla.numpy()  # 获取概率最大处所对应的索引值
        top1_index = get_max(5, predict, predict_cla)
        print(top1_index)

    return top1_index

def print_error(value):
    print("error: ", value)


if __name__ == '__main__':

    if sys.argv[1]=="vgg19":
        model = torchvision.models.vgg19(pretrained=True)
    elif sys.argv[1]=="inception_v3":
        model = torchvision.models.inception_v3(pretrained=True)
    elif sys.argv[1]=="densenet121":
        model = torchvision.models.densenet121(pretrained=True)

    # model = torchvision.models.vgg19(pretrained=True)
    model = model.cuda()
    
    # centerpointsnumber, initnumber, fitnumber
    iter_nums = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))


    avali_count = 0
    
    total_count = 0
    
    cls_count = 1
    
    suc_count = 0
    
    queries = 0

    cls_list = sorted(os.listdir('./ILSVRC2012_img_val_subset'), key=lambda x: int(x))
    for cls in cls_list[:5]:
        count = 1
        if os.path.exists("./tmpimg"):
            shutil.rmtree("./tmpimg")
            os.mkdir("./tmpimg")
        for img_name in os.listdir('./ILSVRC2012_img_val_subset/'+cls):
            cls_index = int(cls)
            img = Image.open('./ILSVRC2012_img_val_subset/'+cls+'/'+img_name).convert('RGB')
            pidx = predict(model, img)
            if int(pidx[0]) == int(cls):
                avali_count = avali_count + 1
                count+=1
                total_count+=1
            else:
                count+=1
                total_count+=1
                continue
                
            print('\nclass:'+str(cls_count)+'/1000————img:'+str(count)+'/5 '+'queries:'+str(queries)+" "+cls)
            x, y, width, rotate, query = BC(img, cls_index, model, './ILSVRC2012_img_val_subset/'+cls+'/'+img_name,img_name,iter_nums)
            queries += query
            img_simulated_path = img_simulate(int(x), int(y), width, rotate, './ILSVRC2012_img_val_subset/'+cls+'/'+img_name)
            img_simulated = Image.open(img_simulated_path).convert("RGB")
            pred_idx = predict(model, img_simulated)
            if int(pred_idx[0]) != int(cls):
                suc_count+=1
        cls_count += 1
    
    with open("./res_log.txt", "w") as fp:
        fp.write("success_rate:"+str(suc_count/avali_count)+" count "+str(suc_count) + " avg_queries: "+str(queries/avali_count))
    print("success_rate:"+str(suc_count/avali_count)+" \nsuc_count "+str(suc_count) + " \navg_queries: "+str(queries/avali_count))
    print("avali_count/total_count: ", avali_count, "/", total_count)
    
    




