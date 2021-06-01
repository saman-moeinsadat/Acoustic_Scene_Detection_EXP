import pandas as pd
import os
from pathlib import Path
import numpy as np
import librosa
from train import model_prepare
import torch
from util import wave_to_array
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


def audio_search(
    path, weight_path, acc, sr=48000, hop_length_in_sec=(0.4, 0.6, 0.8),
    block_lengths=(1, 1.5, 2),
    hop_length=512
):

    root_path = (Path(__file__).parent).resolve()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model_prepare(27, pretrained=False)
    model.load_state_dict(torch.load(
        weight_path
    ))
    model = model.to(device)
    model.eval()

    data = pd.read_csv(path, index_col=0)
    num_iter = 0
    for index, row in data.iterrows():

        print('Iteration: %d/%d' % (num_iter, len(data) - 1))
        num_iter += 1
        audio_series, sample_rate = librosa.load(
            "%s/test/%s.flac" % (root_path, row.name), sr=sr
        )
        flag = False
        for i, block_length in enumerate(block_lengths):
            n_frames = int(5 * sr / hop_length) + 1

            win_length = int(sample_rate * block_length)

            n_last = len(audio_series) % win_length
            audio_series = np.pad(
                audio_series, (0, win_length - n_last),
                'constant', constant_values=(0, 0)
            )

            block_hop_length = int(hop_length_in_sec[i] * sr)
            number_of_blocks = int(
                len(audio_series) / (block_hop_length)
            ) - 1

            df_tensor = np.zeros(
                shape=(number_of_blocks, 1, 128, n_frames)
            )
            for idx in range(number_of_blocks):
                sub_serie = np.array(
                    audio_series[
                        idx*block_hop_length: (idx*block_hop_length)+win_length
                    ]
                )

                data_coded = np.array(
                    [wave_to_array(sub_serie, sample_length=5)]
                )
                df_tensor[idx, :, :, : data_coded.shape[2]] = data_coded
            df_tensor = torch.from_numpy(df_tensor).float()
            dataset = TensorDataset(df_tensor)
            input_loader = DataLoader(
                dataset, batch_size=16, shuffle=False,
                drop_last=False
            )
            for input in input_loader:
                input[0] = input[0].to(device)
                with torch.no_grad():
                    outputs = model(input[0])
                    _, preds = torch.max(outputs, 1)
                if not flag:
                    detections = preds
                    flag = True
                else:
                    detections = torch.cat((detections, preds), 0)
        detections = detections.cpu().tolist()
        detections = list(set([item[0] for item in detections]))
        sample_prediction = [0.00] * 27
        for item in detections:
            sample_prediction[int(item)] = 1.00
        sample_prediction[17] = max(sample_prediction[17], sample_prediction[18])
        sample_prediction[24] = max(sample_prediction[24], sample_prediction[25])
        del sample_prediction[26]
        del sample_prediction[25]
        del sample_prediction[18]
        data.loc[row.name] = sample_prediction

    data.to_csv("%s/results_%.2f.csv" % (root_path, acc))


if __name__ == "__main__":
    PATH = str((Path(__file__).parent).resolve())
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # audio_search("%s/sample_submission.csv" % PATH, "%s/weights/semi/model_semi_weights_97.44_0.pt" % PATH, 97.44)
    # audio_search("%s/sample_submission.csv" % PATH, "%s/weights/semi/model_semi_weights_97.07_0.pt" % PATH, 97.07)
    # audio_search("%s/sample_submission.csv" % PATH, "%s/weights/app/model_semi_weights_98.08_141.pt" % PATH, 98.08)
