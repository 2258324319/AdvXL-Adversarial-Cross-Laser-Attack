import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

def predict():
    trans = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open('E:/PaperCode/Advlight-main/VGG19/testjpg/113_simulated.jpg')
    img = trans(img)
    # 加一个batch维度
    img = torch.unsqueeze(img, dim=0)

    model = torchvision.models.vgg19()
    model.load_state_dict(torch.load('E:/PaperCode/Advlight-main/VGG19/models/vgg19-dcbb9e9d.pth', map_location="cpu"))
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)  # 得到概率分布
        predict_cla = torch.argmax(predict).numpy()  # 获取概率最大处所对应的索引值

        # street sign的索引值919找到置信度
        credit = predict[951]
        # lemon 951
    get_max(5, predict)
    return credit

'''
    获取前几个可能
'''


def get_max(n, pre):
    # 得到从大到小排序的索引号
    pre = np.argsort(-pre)
    # 读取索引对应
    with open('imagenet1000_clsidx_to_labels.txt', 'r') as f:
        line = f.readlines()
    name = []
    for i in range(n):
        print(pre[i])
        name.append(line[int(pre[i])].split('\'')[1])
    return name

#
# print(get_max(5, predict))
predict()