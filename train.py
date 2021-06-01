import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch
import torch.nn as nn
import time
import copy
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.autograd import Variable
from sklearn.metrics import f1_score, recall_score, precision_score, label_ranking_average_precision_score
from sklearn.metrics import accuracy_score
import torchvision.models as models
from util import extract_code_label
from pathlib import Path
import os


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_val(criterion, train_dataset, val_dataset, mode, epochs_number=40, lr=0.0001, bs=16):

    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True
    )
    val_data_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)

    data_loaders = dict(
        zip(['train', 'val'], [train_data_loader, val_data_loader])
    )
    datasets = dict(zip(['train', 'val'], [train_dataset, val_dataset]))

    # preparing the model:
    if mode == 'supervised':
        model = model_prepare(27, pretrained=False)
    # if mode == 'active':
    #     model = model_prepare(26, pretrained=False)
    #     model.load_state_dict(torch.load(
    #         "%s/new_split/weights/model_supervised_weights89.64acc.pt" %\
    #         str((Path(__file__).parent).resolve())
    #     ))
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Reducing learning-rate after a specific epoch:
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[round(epochs_number*x) for x in [0.75, 1]],
        gamma=0.1
    )
    scheduler.last_epoch = epochs_number

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # query_out = 16
    for epoch in range(epochs_number):
        t1 = time.time()
        print('Epoc {}/{}'.format(epoch, epochs_number - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            flag = False
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            for inputs, labels in data_loaders[phase]:
                # if (mode == 'active' and phase == 'train'):
                #
                #     inputs, labels = entropy_sampling(
                #         model, inputs, labels, device, bs_in=bs, bs_out=query_out
                #     )

                # Using torch's Variable for closing the autograd
                # route.
                # labels_expanded = torch.zeros(len(inputs), 26, dtype=torch.float64)
                # for item in range(len(inputs)):
                #     labels_expanded[item, labels[item].item()] = 1.00
                # labels_expanded = Variable(labels_expanded)
                inputs = Variable(inputs.float(), requires_grad=True)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # m = nn.Sigmoid()
                    # outputs = m(outputs).double()

                    # Here the detections are stacked for
                    # measuring the metrics
                    if not flag:
                        # outputs_all = outputs
                        detections = preds
                        labels_all = labels
                        # labels_expanded_all = labels_expanded
                        flag = True
                    else:
                        # outputs_all = torch.cat((outputs_all, outputs), 0)
                        detections = torch.cat((detections, preds), 0)
                        labels_all = torch.cat((labels_all, labels), 0)
                        # labels_expanded_all = torch.cat((labels_expanded_all, labels_expanded), 0)

                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        clip_gradient(model, 0.5)
                        optimizer.step()
                running_loss += loss.item() * inputs.size()[0]

            epoch_loss = running_loss / len(datasets[phase])
            # labels_all, detections,  = labels_all.cpu(), detections.cpu()
            epoch_f1 = f1_score(labels_all, detections, average='weighted')
            epoch_recall = recall_score(
                labels_all, detections, average='weighted'
            )
            epoch_precision = precision_score(
                labels_all, detections, average='weighted'
            )
            epoch_acc = accuracy_score(
                labels_all, detections, normalize=True
            )
            # if phase == 'train' and epoch % 5 == 0 and epoch > 0:
            #     query_out -= 1
            print("""
                {}  ==>  Loss: {:.4f}   Accuracy: {:.2f} %   Recall: {:.2f} %
                         Precision: {:.2f} %   F1_score: {:.2f} %
                """.format(
                phase.title(), epoch_loss, epoch_acc * 100, epoch_recall * 100,
                epoch_precision * 100, epoch_f1 * 100
            ))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val' and epoch_acc*100 >= 95.50:
                torch.save(
                    model.state_dict(), '%s/new_split/weights/new/model_supervised_weights_%.2f_%d.pt' %\
                    (str((Path(__file__).parent).resolve()), epoch_acc*100, epoch)
                )
        scheduler.step()
        t2 = time.time()
        print('Epoch running time in {:.0f}m {:.0f}s'.format(
            (t2 - t1) // 60, (t2 - t1) % 60))
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:.2f} %'.format(best_acc * 100))
    model.load_state_dict(best_model_wts)

    return model


def model_prepare(number_of_classes, pretrained=False):
    model = models.resnet34(pretrained=pretrained)
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )

    model.fc = nn.Linear(
        in_features=512, out_features=number_of_classes, bias=True
    )
    # ct = 0
    # for child in model.children():
    #     if ct < 7:
    #         for param in child.parameters():
    #             param.requires_grad = False
    return model


def prepare_dataset_dataloader(
    dataset, bs=16, val_train_ratio=0.7
):
    train_length = int(val_train_ratio * len(dataset))
    val_length = len(dataset) - train_length
    train_dataset, val_dataset = random_split(
        dataset, [train_length, val_length]
    )
    train_data_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=bs, shuffle=True)
    return train_dataset, val_dataset, train_data_loader, val_data_loader


def entropy_sampling(model, inputs, labels, device, bs_in=64, bs_out=16):
    model.eval()
    outputs = model(inputs.to(device))
    m = nn.Sigmoid()
    outputs = m(outputs)
    entropies = torch.tensor(
        [0 - torch.sum(item*torch.log10(item)).item() for item in outputs],
        dtype=torch.float64
    )
    selected_idx = torch.argsort(entropies, descending=True)[: bs_out].tolist()
    inputs = torch.tensor(
        [inputs[idx].tolist() for idx in selected_idx],
        dtype=torch.float64
    )
    labels = torch.tensor(
        [labels[idx].tolist() for idx in selected_idx],
        dtype=torch.int64
    )
    model.train()
    return inputs, labels


def train_val_fp(
    data_super, data_fp, data_val, epochs_number=25, num_class=26,
):
    PATH = str((Path(__file__).parent).resolve())
    since = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataloader = DataLoader(
        data_fp, batch_size=16, shuffle=True, drop_last=False
    )

    val_data_loader = DataLoader(
        data_val, batch_size=16, shuffle=True, drop_last=False
    )

    super_data_loader = DataLoader(
        data_super, batch_size=16, shuffle=True, drop_last=False
    )

    model = model_prepare(num_class, pretrained=False)
    model.load_state_dict(torch.load(
        "%s/weights/model_supervised_weights_97.67_20.pt" % PATH,
        map_location=device
    ))
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[round(epochs_number*x) for x in [0.3, 1]],
        gamma=0.1
    )
    scheduler.last_epoch = epochs_number
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs_number):
        t1 = time.time()
        print('Epoc {}/{}'.format(epoch, epochs_number - 1))
        print('-' * 10)
        for input_fp, label_fp in dataloader:

            # Forward Pass to get the pseudo labels.
            label_fp = Variable(label_fp)
            input_fp = Variable(
                input_fp, requires_grad=True
            )
            input_fp = input_fp.to(device)
            label_fp = label_fp.to(device)
            model.eval()
            output = model(input_fp)
            m = nn.Sigmoid()
            output = m(output).double()
            # print(label_fp)
            # print(output)
            for item in range(len(output)):
                output[item, label_fp[item].item()] = 0.00
            
            # print(output)


            model.train()
            output_pseudo = model(input_fp)
            m = nn.Sigmoid()
            output_pseudo = m(output_pseudo).double()
            unlabeled_loss = criterion(
                output_pseudo, output.detach()
            )
            optimizer.zero_grad()
            unlabeled_loss.backward()
            clip_gradient(model, 0.5)
            optimizer.step()
        
        for inputs, labels in super_data_loader:
            model.train()
            inputs = Variable(inputs.float(), requires_grad=True)
            inputs, labels = inputs.to(device), labels.to(device)
            labels_expanded = torch.zeros(len(inputs), 26, dtype=torch.float64)
            for item in range(len(inputs)):
                labels_expanded[item, labels[item].item()] = 1.00
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                m = nn.Sigmoid()
                outputs = model(inputs)
                outputs = m(outputs).double()
                loss = criterion(outputs, labels_expanded)
                loss.backward()
                clip_gradient(model, 0.5)
                optimizer.step()
        

        running_loss = 0.0
        model.eval()
        flag = False
        with torch.no_grad():
            
            for inputs, labels in val_data_loader:
                labels = Variable(labels)
                inputs = Variable(inputs.float(), requires_grad=False)
                inputs, labels = inputs.to(device), labels.to(device)

                labels_expanded = torch.zeros(len(inputs), 26, dtype=torch.float64)
                for item in range(len(inputs)):
                    labels_expanded[item, labels[item].item()] = 1.00
                
                m = nn.Sigmoid()

                outputs = model(inputs)
                outputs = m(outputs).double()

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels_expanded)

                if not flag:

                    detections = preds
                    labels_all = labels
                    flag = True
                else:

                    detections = torch.cat((detections, preds), 0)
                    labels_all = torch.cat((labels_all, labels), 0)

                
                running_loss += loss.item() * inputs.size()[0]
        epoch_loss = running_loss / len(data_val)
        labels_all = labels_all.cpu()
        detections = detections.cpu()
        epoch_f1 = f1_score(labels_all, detections, average='weighted')
        epoch_recall = recall_score(
            labels_all, detections, average='weighted'
        )
        epoch_precision = precision_score(
            labels_all, detections, average='weighted'
        )
        epoch_acc = accuracy_score(
            labels_all, detections, normalize=True)

        print("""
            Validation  ==>  Loss: {:.6f}   Recall: {:.2f} %
            Precision: {:.2f} %   F1_score: {:.2f} %  Accuracy: {:.2f} %
        """.format(
            epoch_loss, epoch_recall * 100,
            epoch_precision * 100, epoch_f1 * 100, epoch_acc * 100
        ))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        if epoch_acc*100 >= 96.50:
            torch.save(
                model.state_dict(), '%s/weights/fp/model_semi_weights_%.2f_%d.pt' %\
                (PATH, epoch_acc*100, epoch)
            )
        model.train()
        t2 = time.time()
        scheduler.step()
        print('Epoch running time:  {:.0f}m {:.0f}s'.format(
            (t2 - t1) // 60, (t2 - t1) % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:.2f} %'.format(best_acc * 100))
    model.load_state_dict(best_model_wts)

    return model


if __name__ == "__main__":
    PATH = str((Path(__file__).parent).resolve())
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # data_super = torch.from_numpy(np.load("%s/new_split/data_train_supervised.npy" % PATH)).float()
    # label_super, _ = extract_code_label("%s/new_split/label_train_supervised.npy" % PATH)
    # label_super = torch.from_numpy(label_super)
    # dataset_super = TensorDataset(data_super, label_super)

    data_val = torch.from_numpy(np.load("%s/data_val.npy" % PATH)).float()
    label_val, _ = extract_code_label("%s/label_val.npy" % PATH)
    label_val = torch.from_numpy(label_val)
    dataset_val = TensorDataset(data_val, label_val)

    # best_model_super = train_val(
    #     criterion, dataset_super, dataset_val, mode='supervised'
    # )
    # torch.save(best_model_super.state_dict(), PATH+'/new_split/weights/model_supervised_weights.pt')

    data_super = torch.from_numpy(np.load("%s/data_train_85.npy" % PATH)).float()
    label_super, _ = extract_code_label("%s/label_train_85.npy" % PATH)
    label_super = torch.from_numpy(label_super)
    dataset_super = TensorDataset(data_super, label_super)

    # weights = torch.FloatTensor([
    #     85 / len(os.listdir("%s/new_split/data_train_all/%s" % (PATH, dir)))\
    #     for dir in sorted(
    #         os.listdir("%s/new_split/data_train_all/" % PATH),
    #         key=lambda x: float(x.replace("-", "."))
    #     )
    # ])

    # print(weights)

    # criterion = nn.CrossEntropyLoss()
    # print(weights)

    #best_model_super = train_val(
    #    criterion, dataset_super, dataset_val, mode='supervised', epochs_number=60
    #)
    #torch.save(best_model_super.state_dict(), PATH+'/new_split/weights/new/model_supervised_weights_85.pt')

    # data_active = torch.from_numpy(np.load("%s/new_split/data_active.npy" % PATH)).float()
    # label_active, _ = extract_code_label("%s/new_split/label_active.npy" % PATH)
    # label_active = torch.from_numpy(label_active)
    # dataset_active = TensorDataset(data_active, label_active)
    # best_model_active = train_val(
    #      criterion, dataset_active, dataset_val, mode='active',
    #      epochs_number=60, lr=0.001, bs=32
    # )
    # torch.save(best_model_active.state_dict(), PATH+'/new_split/weights/model_active_weights.pt')
    data_fp = torch.from_numpy(np.load("%s/data_fp.npy" % PATH)).float()
    label_fp, _ = extract_code_label("%s/label_fp.npy" % PATH)
    label_fp = torch.from_numpy(label_fp)
    dataset_fp = TensorDataset(data_fp, label_fp)
    best_model_fp = train_val_fp(
        dataset_super, dataset_fp, dataset_val
    )
    torch.save(best_model_fp.state_dict(), PATH+'/weights/fp/model_semi_weights.pt')
