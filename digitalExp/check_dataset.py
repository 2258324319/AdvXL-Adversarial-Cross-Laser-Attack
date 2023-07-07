import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import shutil


def evalution():
    model = torchvision.models.vgg19(pretrained=True)
    model = model.to('cuda:0')
    count = 0
    true_count = 0
    res_dict = {}
    
    for cls in os.listdir("./ILSVRC2012_img_val_subset/"):
        print(count)
        for img_name in os.listdir("./ILSVRC2012_img_val_subset/"+cls+"/"):
            count+=1
            img = Image.open("./ILSVRC2012_img_val_subset/"+cls+"/"+img_name).convert('RGB')
            # 这一串是为了把Image对象变成和读取文件一样的JPEGImagePIL对象，才不会出错
            top1_pre_index = predict(model, img)
            for tidx in top1_pre_index:
                if str(tidx) not in res_dict:
                    res_dict[str(tidx)] = 1
                else:
                    res_dict[str(tidx)] += 1
            
            if int(cls) in top1_pre_index:
                true_count+=1

    print('count',count)
    print('true_count', true_count)
    print('pre_res', res_dict)
    

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

    return top1_index




evalution()


