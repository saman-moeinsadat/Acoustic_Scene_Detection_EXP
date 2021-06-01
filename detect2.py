import pandas as pd
from pathlib import Path
import numpy as np
import librosa
from train import model_prepare
import torch
from util import wave_to_array
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def audio_search(
        path, weights_path, acc, sr=48000
):

    root_path = (Path(__file__).parent).resolve()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_prepare(26, pretrained=False)
    model = model.to(device)
    model.load_state_dict(torch.load(
        weights_path,
        map_location= 'cpu'
    ))
    model.eval()
    print("model loaded!")
    m = nn.Softmax(dim=1)

    win_lens = [
        0.67, 1.02, 1.14, 1.3, 1.52, 1.69, 1.98, 2.22, 2.53, 2.8, 3.0, 3.21,
        3.54, 3.77, 4.0, 4.52, 5.78, 6.48, 6.85, 8.32
    ]
    data = pd.read_csv(path, index_col=0)
    num_iter = 0
    for index, row in data.iterrows():

        

        print('Iteration: %d/%d' % (num_iter, len(data) - 1))
        
        # if num_iter <= 700:
        #     num_iter += 1
        #     continue
        
        audio_series, sample_rate = librosa.load(
            "%s/test/%s.flac" % (root_path, row.name), sr=sr
        )
        detections = []
        for i, win_len in enumerate (win_lens):
            win_len = int(win_len * sr)
            n_last = len(audio_series) % win_len
            audio_series = np.pad(
                audio_series, (0, win_len - n_last),
                'constant', constant_values=(0, 0)
            )
            win_step = int(win_len * 0.5)
        
            number_of_blocks = int(
                len(audio_series) / (win_step)
            ) - 1

            df_tensor = np.zeros(
                shape=(number_of_blocks, 1, 224, 224)
            )
            for idx in range(number_of_blocks):
                sub_serie = np.array(
                    audio_series[
                        idx*win_step: (idx*win_step)+win_len
                    ]
                )
                data_coded = np.array(
                        [wave_to_array(sub_serie, sample_rate=sr)]
                )
                data_coded = np.resize(data_coded, (1, 224, 224))
                df_tensor[idx, :, :, :] = data_coded
            df_tensor = torch.from_numpy(df_tensor).float()
            dataset = TensorDataset(df_tensor)
            input_loader = DataLoader(
                dataset, batch_size=64, shuffle=False,
                drop_last=False
            )
            for input in input_loader:
                input[0] = input[0].to(device)
                with torch.no_grad():
                    outputs = model(input[0])
                    _, preds = torch.max(outputs, 1)
                    outputs = m(outputs).cpu()
                for item in range(len(input[0])):
                    highconf_item = outputs[item, preds[item].item()].item()
                #print(highconf_item, preds[item].item())
                    if highconf_item >= 0.99:
                        detections.append(preds[item].item())
        
        detections = list(set(detections))
        print("Detection: ",detections)
        sample_prediction = [0.00] * 26
        for item in detections:
            sample_prediction[int(item)] = 1.00
        sample_prediction[17] = max(sample_prediction[17], sample_prediction[18])
        sample_prediction[24] = max(sample_prediction[24], sample_prediction[25])
        del sample_prediction[25]
        del sample_prediction[18]
        data.loc[row.name] = sample_prediction
        print(sample_prediction)
        print("Recording_id: ", row.name)
        # if num_iter % 50 == 0:
        #     data.to_csv("%s/results/results_%d.csv" % (root_path, num_iter))
        num_iter += 1
        print("*************************")
        
        


    data.to_csv("%s/results_%.2f.csv" % (root_path, acc))


if __name__ == "__main__":
    PATH = str((Path(__file__).parent).resolve())
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    audio_search("%s/sample_submission.csv" % PATH, "%s/weights/fp/96.12acc.pt" % PATH, 96.12)
    #audio_search("%s/sample_submission.csv" % PATH, "%s/weights/semi/model_semi_weights_97.07_0.pt" % PATH, 97.07)
    # audio_search("%s/sample_submission.csv" % PATH, "%s/weights/app/model_semi_weights_98.08_141.pt" % PATH, 98.08)
