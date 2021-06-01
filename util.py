import pandas as pd
from pathlib import Path
import os
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, convolve
from shutil import copyfile, rmtree
from math import sqrt
import random
from numpy.random import randint
import librosa.effects


def dataset_build(path, sr=48000):

    n_true, n_false = 0, 0
    data_true = pd.read_csv("%s/train_tp.csv" % path.parent)
    data_false = pd.read_csv("%s/train_fp.csv" % path.parent)
    os.mkdir("%s/new_split/data_false/" % path)
    # os.mkdir("%s/new_split/data_true/" % path)

    for item in os.listdir("%s/train/" % path.parent):
        audio_series, sample_rate = librosa.load(
            "%s/train/%s" % (path.parent, item), sr=sr
        )
        audio_series = np.pad(
            audio_series, (0, 9600), 'constant', constant_values=(0, 0)
        )
        rec_id = item.split('.')[0]
        # for idx, fts in data_true.loc[data_true.recording_id == rec_id].iterrows():
        #     instance_sample = audio_series[
        #         round(fts.t_min*sr)-9600: round(fts.t_max*sr)+9600
        #     ]
        #     if not os.path.isdir(
        #         "%s/new_split/data_true/%s-%s" % (path, fts.species_id, fts.songtype_id)
        #     ):
        #         os.mkdir("%s/new_split/data_true/%s-%s" % (
        #             path, fts.species_id, fts.songtype_id)
        #         )
        #     sf.write(
        #         '%s/new_split/data_true/%s-%s/%d-%s-%s.wav' % (
        #             path, fts.species_id, fts.songtype_id, n_true,
        #             fts.f_min, fts.f_max
        #         ),
        #         instance_sample, sr, subtype='PCM_24'
        #         )
        #     n_true += 1
        for idx, fts in data_false.loc[data_false.recording_id == rec_id].iterrows():
            instance_sample = audio_series[
                round(fts.t_min*sr)-9600: round(fts.t_max*sr)+9600
            ]
            if not os.path.isdir("%s/new_split/data_false/%s-%s" % (
                path, fts.species_id, fts.songtype_id
            )):
                os.mkdir("%s/new_split/data_false/%s-%s" % (
                    path, fts.species_id, fts.songtype_id
                ))
            sf.write(
                '%s/new_split/data_false/%s-%s/%d-%s-%s.wav' % (
                    path, fts.species_id, fts.songtype_id, n_false,
                    fts.f_min, fts.f_max
                ),
                instance_sample, sr, subtype='PCM_24'
                )
            n_false += 1


def butter_bandpass(lowcut, highcut, sr=48000, order=21):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    butter_sos = butter(order, [low, high], btype='band', output='sos')
    return butter_sos


def butter_bandpass_filter(data, lowcut, highcut, sr=48000, order=21):
    butter_sos = butter_bandpass(lowcut, highcut, sr, order=order)
    y = sosfilt(butter_sos, data)
    return y


def half_wave(path, sr=48000, length=4):
    length_limit = length * sr

    for item in os.listdir("%s/new_split/data_train/" % (path)):
        for file in os.listdir("%s/new_split/data_train/%s/" % (path, item)):
            audio_series, sample_rate = librosa.load(
                "%s/new_split/data_train/%s/%s" % (path, item, file), sr=sr
            )
            if len(audio_series) > length_limit:
                if not os.path.isdir("%s/new_split/data_train/%s_extras/" % (path, item)):
                    os.mkdir("%s/new_split/data_train/%s_extras/" % (path, item))

                first_half = int(len(audio_series) / 2)

                sf.write(
                    '%s/new_split/data_train/%s_extras/%s.1.wav' % (path, item, file),
                    audio_series[: first_half], sr, subtype='PCM_24'
                )
                sf.write(
                    '%s/new_split/data_train/%s_extras/%s.2.wav' % (path, item, file),
                    audio_series[first_half:], sr, subtype='PCM_24'
                )


def gain(path, gain_db, sr=48000, keep_original=False):
    parent_dir = str(Path(path).parent)
    new_name = path.split("/")[-1][: -3] + "amp.wav"
    time_series, sample_rate = librosa.load(path, sr=sr)
    amp_factor = sqrt(pow(10, gain_db/10))
    sf.write(
        "%s/%s" % (parent_dir, new_name),
        amp_factor*time_series, sr, subtype='PCM_24'
    )
    if not keep_original:
        os.remove(path)


def data_augmentation(path, desired_num=85, sr=48000):
    class_names = sorted(os.listdir("%s/new_split/data_train/" % path))
    stretches = [0.6, 1.4, 0.4]
    impulse_responses = [
        librosa.load("%s/IRs/%s" % (path, file), sr=sr)[0] for file\
        in os.listdir("%s/IRs" % path)
    ]
    pitches = [-6, 6, 9]

    os.mkdir("%s/new_split/data_true_augmented/" % path)
    for cls in class_names:
        n = 0
        os.mkdir("%s/new_split/data_true_augmented/%s" % (path, cls))
        cls_data_true = os.listdir("%s/new_split/data_train/%s" % (path, cls))
        res = randint(
            0, len(cls_data_true), int(desired_num) - len(cls_data_true)
        )
        win_len = int(len(res)/3)
        res = [
            res[k*win_len: (k+1)*win_len] if k != 2 else\
            res[k*win_len:] for k in range(3)
        ]

        stretch_dict = dict()
        for file_idx in res[0]:
            if file_idx not in stretch_dict:
                stretch_dict[file_idx] = []
            audio_series, sample_rate = librosa.load(
                '%s/new_split/data_train/%s/%s' % (path, cls, cls_data_true[file_idx]),
                sr=sr
            )
            if len(stretch_dict[file_idx]) == 0:
                stretch_item = stretches[0]
                stretch_dict[file_idx].append(stretches[0])
            elif len(stretch_dict[file_idx]) == 1:
                stretch_item = stretches[1]
                stretch_dict[file_idx].append(stretches[1])
            elif len(stretch_dict[file_idx]) == 2:
                stretch_item = stretches[2]
                stretch_dict[file_idx].append(stretches[2])
            elif len(stretch_dict[file_idx]) == 3:
                continue
            audio_augmented = librosa.effects.time_stretch(
                audio_series,
                stretch_item
            )
            new_name = cls_data_true[file_idx][: -3] + "%d_streched.wav" % n
            sf.write(
                    '%s/new_split/data_true_augmented/%s/%s' % (path, cls, new_name),
                    audio_augmented, sr, subtype='PCM_24'
                    )
            n += 1

        convolved_dict = dict()
        for file_idx in res[1]:
            if file_idx not in convolved_dict:
                convolved_dict[file_idx] = []
            audio_series, sample_rate = librosa.load(
                '%s/new_split/data_train/%s/%s' %\
                (path, cls, cls_data_true[file_idx]),
                sr=sr
            )
            if len(convolved_dict[file_idx]) == 0:
                convolve_item = impulse_responses[0]
                convolved_dict[file_idx].append(0)
            elif len(convolved_dict[file_idx]) == 1:
                convolve_item = impulse_responses[1]
                convolved_dict[file_idx].append(1)
            elif len(convolved_dict[file_idx]) == 2:
                continue
            new_name = cls_data_true[file_idx][: -3] + "%d_convolved.wav" % n
            audio_augmented = convolve(
                audio_series,
                convolve_item, mode='same'
            )
            sf.write(
                    '%s/new_split/data_true_augmented/%s/%s' % (path, cls, new_name),
                    audio_augmented, sr, subtype='PCM_24'
                    )
            n += 1

        pitch_dict = dict()
        for file_idx in res[2]:
            if file_idx not in pitch_dict:
                pitch_dict[file_idx] = []
            audio_series, sample_rate = librosa.load(
                '%s/new_split/data_train/%s/%s' % (path, cls, cls_data_true[file_idx]),
                sr=sr
            )
            if len(pitch_dict[file_idx]) == 0:
                pitch_item = pitches[0]
                pitch_dict[file_idx].append(0)
            elif len(pitch_dict[file_idx]) == 1:
                pitch_item = pitches[1]
                pitch_dict[file_idx].append(1)
            elif len(pitch_dict[file_idx]) == 2:
                pitch_item = pitches[2]
                pitch_dict[file_idx].append(2)
            elif len(pitch_dict[file_idx]) == 3:
                continue
            audio_augmented = librosa.effects.pitch_shift(
                audio_series, sample_rate,
                pitch_item,
                bins_per_octave=24
            )
            new_name = cls_data_true[file_idx][: -3] + "%d_pitched.wav" % n
            sf.write(
                    '%s/new_split/data_true_augmented/%s/%s' % (path, cls, new_name),
                    audio_augmented, sr, subtype='PCM_24'
                    )
            n += 1


def dataset_clean_number(path):
    dir_list = sorted(
        os.listdir(path),
        key=lambda x: float(x.replace("-", "."))
    )
    dataset_length = 0
    for dir in dir_list:
        sub_dir = os.listdir(path+'/'+dir)
        for file in sub_dir:
            if file.endswith(".wav"):
                dataset_length += 1
            else:
                # Removes the unwanted files.
                os.remove(path+'/'+dir+'/'+file)
    return dataset_length


def dir_to_df(
    path, dst_length, name, n_mels=128,
    sample_rate=48000, hop_length=512, frame_length=5
):

    dir_list = sorted(
        os.listdir(path),
        key=lambda x: float(x.replace("-", "."))
    )
    label = []

    PATH = (Path(__file__).parent).resolve()

    # numpy empy array with shape:
    # (dataset-length, 1, number of mel features, number of frames)

    n_frames = int(frame_length * sample_rate / hop_length) + 1
    df_tensor = np.zeros(shape=(dst_length, 1, 224, 224))

    n = 0
    for item in dir_list:
        sub_dir_list = os.listdir('%s/%s' % (path, item))
        for sub_item in sub_dir_list:
            if sub_item.endswith(".wav"):
                label.append(item)
                data_coded = np.array(
                    [wave_to_array('%s/%s/%s' % (path, item, sub_item))]
                )
                # print(data_coded.shape)
                data_coded = np.resize(data_coded, (1, 224, 224))
                df_tensor[n, :, :, :] = data_coded
                n += 1

    label = np.array(label)
    np.save('/%s/data_%s.npy' % (PATH, name), df_tensor)
    np.save('/%s/label_%s.npy' % (PATH, name), label)


def wave_to_array(data, sample_rate=48000, n_mels=128, sample_length=5):
    # Checking the input and read the data.

    if isinstance(data, str):
        audio_series, sample_rate = librosa.load(data, sr=sample_rate)
    elif isinstance(data, np.ndarray):
        audio_series = data
    else:
        return """
                The input type must be either path to the
                sound file of numpy ndarray.
                """

    # number_of_samples = len(audio_series)

    # Zero pads the data to have windows-length equal to
    # sample_length.

    # fix_number_of_samples = int(sample_rate*sample_length)
    # if number_of_samples <= fix_number_of_samples:
    #     audio_series = np.pad(audio_series, (
    #         0, fix_number_of_samples - number_of_samples),
    #         'constant', constant_values=(0, 0)
    #     )
    # else:
    #     audio_series = audio_series[:fix_number_of_samples-1]

    # Normalization:

    audio_series_norm = librosa.util.normalize(audio_series)

    # Pre-emphasis:

    audio_series_emp = librosa.effects.preemphasis(audio_series_norm)

    # Mel Spectrogram: extracts 128 Mel features for 259 frames ==>
    # numpy ndarray with (128, 259) shape:

    mel = librosa.feature.melspectrogram(
        audio_series_emp,
        sr=sample_rate,
        n_mels=n_mels
    )

    return librosa.power_to_db(mel, ref=np.max)


def extract_code_label(
    label_path="%s/label_true_supervised.npy" %\
    str((Path(__file__).parent).resolve())
):
    label = np.ndarray.tolist(np.load(label_path))
    cls = sorted(
        list(set(label)),
        key=lambda x: float(x.replace("-", "."))
    )
    for i in range(len(cls)):
        for j in range(len(label)):
            if cls[i] == label[j]:
                label[j] = i
    return np.array(label, dtype=int), np.array(cls)


if __name__ == "__main__":
    PATH = (Path(__file__).parent).resolve()
    # dataset_build(PATH)
    # os.mkdir("%s/new_split/data_train/" % PATH)
    # os.mkdir("%s/new_split/data_val/" % PATH)
    # for cls in os.listdir("%s/new_split/data_true/" % (PATH)):
    #     os.mkdir("%s/new_split/data_train/%s" % (PATH, cls))
    #     os.mkdir("%s/new_split/data_val/%s" % (PATH, cls))
    #     files = os.listdir("%s/new_split/data_true/%s/" % (PATH, cls))
    #     for idx in range(10):
    #         copyfile(
    #             "%s/new_split/data_true/%s/%s" % (PATH, cls, files[idx]),
    #             "%s/new_split/data_val/%s/%s" % (PATH, cls, files[idx])
    #         )
    #     for idx in range(10, len(files)):
    #         copyfile(
    #             "%s/new_split/data_true/%s/%s" % (PATH, cls, files[idx]),
    #             "%s/new_split/data_train/%s/%s" % (PATH, cls, files[idx])
    #         )

    # half_wave(PATH)
    # for item in os.listdir("%s/new_split/data_train/" % (PATH)):
    #     if "extras" in item:
    #         for file in os.listdir("%s/new_split/data_train/%s/" % (PATH, item[: -7])):
    #             if file.split("-")[0] not in [x.split("-")[0] for x in os.listdir(
    #                 "%s/new_split/data_train/%s/" % (PATH, item)
    #             )]:
    #                 copyfile(
    #                     "%s/new_split/data_train/%s/%s" % (PATH, item[: -7], file),
    #                     "%s/new_split/data_train/%s/%s" % (PATH, item, file)
    #                 )
    #         rmtree("%s/new_split/data_train/%s/" % (PATH, item[: -7]))
    #         os.rename(
    #             "%s/new_split/data_train/%s/" % (PATH, item),
    #             "%s/new_split/data_train/%s/" % (PATH, item[: -7])
    #         )

    # data_augmentation(PATH)
    # os.mkdir("%s/new_split/data_train_all/" % PATH)
    # class_names = sorted(os.listdir("%s/new_split/data_train/" % PATH))
    # for cls in class_names:
    #     os.mkdir("%s/new_split/data_train_all/%s" % (PATH, cls))
    #     for dir in ['data_train', 'data_true_augmented']:
    #         for file in os.listdir("%s/new_split/%s/%s" % (PATH, dir, cls)):
    #             copyfile(
    #                 "%s/new_split/%s/%s/%s" % (PATH, dir, cls, file),
    #                 "%s/new_split/data_train_all/%s/%s" % (PATH, cls, file)
    #             )
    # os.mkdir("%s/new_split/super" % PATH)
    # os.mkdir("%s/new_split/active" % PATH)
    # for cls in os.listdir("%s/new_split/data_train_all/" % PATH):
    #     files = os.listdir("%s/new_split/data_train_all/%s/" % (PATH, cls))
    #     len_super = int(0.4 * len(files))
    #     len_active = len(files) - len_super
    #     indices = [idx for idx in range(len(files))]
    #
    #     os.mkdir("%s/new_split/super/%s" % (PATH, cls))
    #     super = random.sample(indices, len_super)
    #     for item in super:
    #         copyfile(
    #             "%s/new_split/data_train_all/%s/%s" % (PATH, cls, files[item]),
    #             "%s/new_split/super/%s/%s" % (PATH, cls, files[item])
    #         )
    #         indices.remove(item)
    #
    #     os.mkdir("%s/new_split/active/%s" % (PATH, cls))
    #     for item in indices:
    #         copyfile(
    #             "%s/new_split/data_train_all/%s/%s" % (PATH, cls, files[item]),
    #             "%s/new_split/active/%s/%s" % (PATH, cls, files[item])
    #         )

    # val_length = dataset_clean_number("%s/new_split/data_val/" % (PATH))
    # dir_to_df(
    #     "%s/new_split/data_val/" % PATH, val_length, 'val'
    # )
    # super_length = dataset_clean_number("%s/new_split/data_train_all/" % (PATH))
    # dir_to_df(
    #     "%s/new_split/data_train_all/" % PATH, super_length, 'train_85'
    # )
    # label, _ = extract_code_label("%s/new_split/label_train_85.npy" % PATH)
    # print(label.tolist())
    # active_length = dataset_clean_number("%s/new_split/active/" % (PATH))
    # dir_to_df(
    #     "%s/new_split/active/" % PATH, active_length, 'active'
    # )

    semi_length = dataset_clean_number("%s/data_false/" % (PATH))
    dir_to_df(
       "%s/data_false/" % PATH, semi_length, 'fp'
    )
