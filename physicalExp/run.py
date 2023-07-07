import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import shutil
import sys


def evalution(direct):
    model = torchvision.models.vgg19(pretrained=True)
    model = model.to('cuda:0')
    count = 0
    false_count = 0
    res_dict = {}

    for img_name in os.listdir('./' + direct + '/'):
        count += 1
        cls_index = 919
        img = Image.open('./' + direct + '/' + img_name).convert('RGB')
        top1_pre_index, score = predict(model, img, 1)
        for tidx in top1_pre_index:
            if str(tidx) not in res_dict:
                res_dict[str(tidx)] = 1
            else:
                res_dict[str(tidx)] += 1

        cls_index = int(cls_index)
        print(img_name, top1_pre_index[0], score[top1_pre_index[0]])
        if cls_index not in top1_pre_index:
            false_count += 1

    print('count', count)
    print('false_count', false_count)
    print('pre_res', res_dict)
    print("ASR: " + str(100 * false_count / count) + "%")


def predict(model, img, n):
    def get_max(n, pre, pre_cla):
        predict = pre
        pre = np.argsort(-pre)
        with open('./imagenet1000_clsidx_to_labels.txt', 'r') as f:
            line = f.readlines()
        topn_index = []
        for i in range(n):
            topn_index.append(int(pre[i])
                              )

        return topn_index

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = trans(img)
    img = img.to('cuda:0')
    img = torch.unsqueeze(img, dim=0)
    model.eval()
    with torch.no_grad():
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict = predict.to('cpu')
        predict_cla = torch.argmax(predict)
        predict_cla = predict_cla.to('cpu')
        predict_cla = predict_cla.numpy()
        topn_index = get_max(n, predict, predict_cla)

    return topn_index, predict

if __name__ == '__main__':
    direct = sys.argv[1]
    evalution(direct)


