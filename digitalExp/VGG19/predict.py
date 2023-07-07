import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def predict_with_vgg19(model, img_path, cls_index, flag=False):
    trans = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = Image.open(img_path)
    print('img_size:',img.size)
    img = trans(img).to('cuda:0')
    # 加一个batch维度
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0) .to('cpu') # 得到概率分布
        predict_cla = torch.argmax(predict).numpy()  # 获取概率最大处所对应的索引值

        # street sign的索引值919找到置信度
        if flag == True:
            print(get_max(5, predict, predict_cla))
       
        credit = predict[cls_index]
    return credit

'''
    获取前几个可能
'''




def get_max(n, pre, pre_cla):
    # 得到从大到小排序的索引号
    predict = pre
    pre = np.argsort(-pre)
    # 读取索引对应
    with open('E:\PaperCode\Advlight-main\VGG19\imagenet1000_clsidx_to_labels.txt', 'r') as f:
        line = f.readlines()
    name = []
    for i in range(n):
        print(pre[i])
        name.append(((line[int(pre[i])].split('\'')[1]), predict[pre[i]]))
    
    return name

#
# print(get_max(5, predict))