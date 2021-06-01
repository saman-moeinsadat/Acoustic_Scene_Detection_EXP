import os
from pathlib import Path
import torchvision.models as models
import torch.nn as nn
import torch
from util import wave_to_array
import numpy as np
from shutil import copyfile
from random import shuffle
from util import extract_code_label
from torch.utils.data import TensorDataset, DataLoader


def model_prepare(number_of_classes, pretrained=False):
    model = models.resnet34(pretrained=pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model.fc = nn.Linear(
    #     in_features=512, out_features=26, bias=True
    # )
    model.fc = nn.Linear(in_features=512, out_features=number_of_classes, bias=True)
    model.load_state_dict(torch.load(
        "%s/new_split/weights/new_26/model_supervised_weights_97.31_33.pt" %\
        str((Path(__file__).parent).resolve()),
        map_location=torch.device('cpu')
    ))
    return model


def no_class_build(path):
    root_path = str((Path(__file__).parent).resolve())
    os.mkdir("%s/new_split/semi/" % root_path)
    os.mkdir("%s/new_split/no_class/" % root_path)

    model = model_prepare(26)
    m = nn.Softmax(dim=1)
    model.eval()

    n_noclass = 0

    for cls in os.listdir(path):
        file_list = os.listdir("%s/%s/" % (path, cls))
        shuffle(file_list)
        cls_count = 0
        os.mkdir("%s/new_split/semi/%s/" % (root_path, cls))
        for file in file_list:
            if n_noclass < 400 and cls_count < 50:
                wave = np.expand_dims(
                    wave_to_array("%s/%s/%s" % (path, cls, file)),
                    axis=0
                )
                wave = torch.unsqueeze(torch.from_numpy(wave).float(), 0)
                with torch.no_grad():
                    output = model(wave)
                _, pred = torch.max(output, 1)
                output = m(output)
                highest_cls_conf = output[0, pred.item()].item()
                print(highest_cls_conf)
                if highest_cls_conf <= 0.80:
                    n_noclass += 1

                    copyfile(
                        "%s/%s/%s" % (path, cls, file),
                        "%s/new_split/no_class/%s" % (root_path, file)
                    )
                else:
                    copyfile(
                        "%s/%s/%s" % (path, cls, file),
                        "%s/new_split/semi/%s/%s" % (root_path, cls, file)
                    )
            elif n_noclass >= 400 or cls_count >= 50:
                copyfile(
                    "%s/%s/%s" % (path, cls, file),
                    "%s/new_split/semi/%s/%s" % (root_path, cls, file)
                )
            cls_count += 1

def find_dist():
    path = str((Path(__file__).parent).resolve())
    data_semi = torch.from_numpy(np.load("%s/data/data_semi.npy" % path)).float()
    label_semi, _ = extract_code_label("%s/data/label_semi.npy" % path)
    label_semi = torch.from_numpy(label_semi)
    dataset_semi = TensorDataset(data_semi, label_semi)
    data_loader = DataLoader(
        dataset_semi, batch_size=16, shuffle=True
    )
    model = model_prepare(27, pretrained=False)
    model.load_state_dict(torch.load(
        "%s/weights/94.91.pt" % path,
        map_location='cpu'
    ))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    flag = False
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            for item in range(len(inputs)):
                outputs[item, labels[item].item()] = -2.00
            _, preds = torch.max(outputs, 1)

            if not flag:
                detections = preds
            else:
                detections = torch.cat((detections, preds), 0)
    print(len(detections))










if __name__ == "__main__":
    # PATH = str((Path(__file__).parent).resolve())
    # no_class_build("%s/new_split/data_false/" % PATH)
    # # list_1 = torch.FloatTensor([-80, 2, 5, 10, -10])
    # # print(list_1)
    # # m = nn.Softmax(dim=0)
    # # list_1 = m(list_1)
    # # print(list_1)
    # find_dist()
    # test_list = [40, 40, 40, 40, 16, 40, 13, 40, 23, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 25, 22, 31, 10, 40, 40, 40, 100]
    # print(sum(test_list))
    # test_tensor = torch.FloatTensor([[0], [1], [2]])
    # new_t = test_tensor.tolist()
    # print([x[0] for x in new_t])
    model = models.resnet34()
    print(model)
