from CUB.models import ModelXtoC
from CUB.config import N_CLASSES, N_ATTRIBUTES
import argparse
import pickle
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import statistics
import pandas as pd
from tqdm import tqdm

# as in CBM
MAPPING = [1, 4, 6, 7, 10, 14, 15, 20, 21, 23, 25, 29, 30, 35, 36, 38, 40, 44, 45, 50, 51, 53, 54, 56, 57, 59, 63, 64, 69, 70, 72, 75, 80, 84, 90, 91, \
    93, 99, 101, 106, 110, 111, 116, 117, 119, 125, 126, 131, 132, 134, 145, 149, 151, 152, 153, 157, 158, 163, 164, 168, 172, 178, 179, 181, \
    183, 187, 188, 193, 194, 196, 198, 202, 203, 208, 209, 211, 212, 213, 218, 220, 221, 225, 235, 236, 238, 239, 240, 242, 243, 244, 249, 253, \
    254, 259, 260, 262, 268, 274, 277, 283, 289, 292, 293, 294, 298, 299, 304, 305, 308, 309, 310, 311]

ATTRIBUTE_FILE = '/path/to/CUB_200_2011/attributes/attributes.txt'
LABEL_PATH = '/path/to/CUB_processed/class_attr_data_10/val.pkl'
IMG_DIR_PATH = '/path/to/CUB_200_2011/images'
N_CLASSES = 200

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--test_for_compliment", action='store_true')
    args = parser.parse_args()
    return args


def get_list_of_used_attributes():
    with open(ATTRIBUTE_FILE, 'r') as f:
        attributes = f.readlines()
    used_attributes = [attributes[m] for m in MAPPING]
    used_attributes = [a.replace('\n', '').split(' ')[1] for a in used_attributes]
    return used_attributes
    
def get_base_attributes(base_class):
    folders = os.listdir(IMG_DIR_PATH)
    for f in folders:
        f_parts = f.split('.')
        if base_class == f_parts[1]:
            base_class_idx = int(f_parts[0]) - 1
    
    data = pickle.load(open(LABEL_PATH, 'rb'))
    for d in data:
        if d['class_label'] == base_class_idx:
            return d['attribute_label']
    raise ValueError('Did not find label for the class: {}'.format(base_class))

def get_a_old(bird, changed_attr, used_attributes):
    attr_type = changed_attr.split('--')[0]
    to_check = [ua for ua in used_attributes if attr_type in ua]
    attr_label = get_base_attributes(bird)
    
    for tc in to_check:
        if attr_label[used_attributes.index(tc)]:
            return used_attributes.index(tc)
    return None

def load_model(args):
    model = torch.load(args.model_path, weights_only=False)
    model.eval()
    return model

def get_img(img_path):
    resol = 299
    transform = transforms.Compose([
        transforms.CenterCrop(resol),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    return img


if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    used_attributes = get_list_of_used_attributes()
    model = load_model(args).to(device)
    stats = []
    stats_nums = []
    total_incorrect = 0
    num_birds = 0

    column_names = ['file', 'changed_attr', 'bird']
    img_paths = pd.read_csv(args.file_path, header=None)
    img_paths.columns = column_names

    for j, img_data in tqdm(img_paths.iterrows(), total=len(img_paths)):
        try:
            attr_changed = img_data['changed_attr']
            idx = used_attributes.index(attr_changed)
        except ValueError:
            continue

        num_birds += 1
        img_path = img_data['file']
        img = get_img(img_path).unsqueeze(dim=0).to(device)
        outputs = model(img)

        sigmoid_outputs = torch.nn.Sigmoid()(torch.cat(outputs, dim=1))
        pred = sigmoid_outputs.cpu() >= 0.5

        if args.test_for_compliment:
            idx = get_a_old(img_path.split('/')[-3], img_data['changed_attr'], used_attributes)
            if idx is None:
                num_birds -= 1
                continue
            if pred[0][idx]:
                total_incorrect += 1
        else:
            if not pred[0][idx]:
                total_incorrect += 1

    with open(args.output_path, mode='w') as output:
        correct = num_birds - total_incorrect
        print('total correct: ' + str(correct / num_birds), file=output)
        print('total incorrect: ' + str(total_incorrect / num_birds), file=output)
        print('images evaluated: {}'.format(num_birds), file=output)
