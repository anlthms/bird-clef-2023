import os
import json
from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.utils.data as data

from util import get_class_names, make_test_augmenters
from dataset import AudioDataset
from models import ModelWrapper
from config import Config

device = torch.device('cpu')
print(f'Running on {device}')

def create_test_loader(conf, input_dir, class_names):
    test_audio_aug, test_image_aug = make_test_augmenters(conf)

    data_dir = 'test_soundscapes'
    test_df = pd.DataFrame()
    data_files = sorted(glob(f'{input_dir}/{data_dir}/*.ogg'))
    assert len(data_files) > 0, f'No files inside {input_dir}/{data_dir}'
    data_files = [os.path.basename(filename) for filename in data_files]
    test_df['filename'] = data_files
    test_df['primary_label'] = class_names[0]
    test_dataset = AudioDataset(
        test_df, conf, input_dir, data_dir,
        class_names, test_audio_aug, test_image_aug, is_test=True)
    print(f'{len(test_dataset)} examples in test set')
    loader = data.DataLoader(
        test_dataset, batch_size=conf.batch_size, shuffle=False,
        num_workers=mp.cpu_count(), pin_memory=False)
    return loader, test_df

def create_model(model_dir, model_file, num_classes):
    checkpoint = torch.load(f'{model_dir}/{model_file}', map_location=device)
    conf = Config(checkpoint['conf'])
    conf.pretrained = False
    model = ModelWrapper(conf, num_classes)
    model = model.to(device)
    model.load_state_dict(checkpoint['model'])
    return model, conf

def predict_batch(outputs, threshold):
    sigmoid = nn.Sigmoid()
    # multi-label classification:
    # set prediction to 1 if probability >= threshold
    return sigmoid(outputs).cpu().numpy() >= threshold

def test(loader, models, num_classes):
    sigmoid = nn.Sigmoid()
    preds = np.zeros((len(models), len(loader.dataset), num_classes), dtype=np.float32)
    start_idx = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            for model_idx, model in enumerate(models):
                model.eval()
                outputs = model(inputs)
                preds[model_idx, start_idx:start_idx + outputs.shape[0]] = sigmoid(outputs).cpu().numpy()
            start_idx += outputs.shape[0]
    return preds.mean(axis=0)

def save_results(conf, input_dir, df, preds, class_names, threshold):
    num_segs = conf.num_segs
    assert preds.shape[0]%num_segs == 0

    # pull predictions towards the mean
    preds = preds.reshape((preds.shape[0]//num_segs, num_segs, -1))
    mean_preds = preds.mean(axis=1, keepdims=True)
    preds = 0.9*preds + 0.1*mean_preds

    # smoothen
    for i in range(num_segs):
        if i == 0:
            preds[:, i] = 0.8*preds[:, i] + 0.2*preds[:, i + 1]
        elif i == num_segs - 1:
            preds[:, i] = 0.2*preds[:, i - 1] + 0.8*preds[:, i]
        else:
            preds[:, i] = 0.1*preds[:, i - 1] + 0.8*preds[:, i] + 0.1*preds[:, i + 1]

    preds = preds.reshape((preds.shape[0]*preds.shape[1], -1))
    class_map = {name: i for i, name in enumerate(class_names)}

    row_ids = []
    targets = []
    for filename in df['filename'].values:
        for clip_idx in range(num_segs):
            end_time = 5*(clip_idx + 1)
            key = f'{filename[:-4]}_{end_time}'
            row_ids.append(key)
    subm = pd.DataFrame()
    subm['row_id'] = row_ids
    assert len(row_ids) == preds.shape[0]
    for i, bird in enumerate(class_names):
        subm[bird] = preds[:, i]
    subm.to_csv('submission.csv', index=False)
    print('Saved submission.csv')

def run(input_dir, model_dir, model_files, threshold):
    meta_file = os.path.join(input_dir, 'train_metadata.csv')
    train_df = pd.read_csv(meta_file, dtype=str)
    class_names = np.array(get_class_names(train_df))
    num_classes = len(class_names)
    models = []
    preds = []
    for model_file in model_files:
        model, conf = create_model(model_dir, model_file, num_classes)
        # assume that conf is the same for all models
        models.append(model)
    loader, df = create_test_loader(conf, input_dir, class_names)
    assert len(loader.dataset) == conf.num_segs*df.shape[0]
    #preds.append(test(loader, model, num_classes))
    #final_preds = np.stack(preds).mean(axis=0)
    final_preds = test(loader, models, num_classes)
    save_results(conf, input_dir, df, final_preds, class_names, threshold)

if __name__ == '__main__':
    test_threshold = 0.04
    run('../input', './', ['model1.pth', 'model2.pth'], test_threshold)
