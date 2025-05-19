import argparse
import pickle
import gzip
import json
import sys
from collections import defaultdict
from tqdm import tqdm
from copy import deepcopy


def get_dataset_paths(dataset, split):
    paths = {
        'phoenix': {
            'iso': f'../../data/phoenix_2014t/phoenix_iso.{split}',
            'iso_blank': f'../../data/phoenix_2014t/phoenix_iso_blank.{split}',
            'ori': f'../../data/phoenix_2014t/phoenix.{split}',
            'bag2items': f'../../data/phoenix_2014t/phoenix_iso_center_label_bag2items.{split}',
            'center_label': f'../../data/phoenix_2014t/phoenix_iso_center_label.{split}',
            'vocab': '../../data/phoenix_2014t/phoenix_iso_with_blank.vocab'
        },
        'csl': {
            'iso': f'../../data/csl-daily/csl_iso.{split}',
            'iso_blank': f'../../data/csl-daily/csl_iso_blank.{split}',
            'ori': f'../../data/csl-daily/csl-daily.{split}',
            'bag2items': f'../../data/csl-daily/csl_iso_center_label_bag2items.{split}',
            'center_label': f'../../data/csl-daily/csl_iso_center_label.{split}',
            'vocab': '../../data/csl-daily/csl_iso_with_blank.vocab'
        },
        'ce_csl': {
            'iso': f'../../data/ce_csl/ce_csl_iso.{split}',
            'iso_blank': f'../../data/ce_csl/ce_csl_iso_blank.{split}',
            'ori': f'../../data/ce_csl/ce_csl.{split}',
            'bag2items': f'../../data/ce_csl/ce_csl_iso_center_label_bag2items.{split}',
            'center_label': f'../../data/ce_csl/ce_csl_iso_center_label.{split}',
            'vocab': '../../data/ce_csl/ce_csl_iso_with_blank.vocab'
        }
    }
    if dataset not in paths:
        raise ValueError(f"Unknown dataset: {dataset}")
    return paths[dataset]


def load_pickle(path, gz=False):
    if gz:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(path, 'rb') as f:
            return pickle.load(f)


def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def build_bag2items(data, ori, win_size=16):
    vfile2items = defaultdict(list)
    for item in data:
        vfile2items[item['video_file']].append(item)
    vfile2len = {item['name']: item['num_frames'] for item in ori}

    bag2items = defaultdict(list)
    bag_idx = 0
    win_size = 16
    for vfile, item_lst in tqdm(vfile2items.items()):
        vlen = vfile2len[vfile]
        for item in item_lst:
            start, end = item['start'], item['end']
            base_start, base_end = start, end
            new_item = deepcopy(item)
            new_item['bag'] = bag_idx
            new_item['aug'] = 0
            new_item['base_start'], new_item['base_end'] = base_start, base_end
            bag2items[str(bag_idx)].append(new_item)
            for cen in range(start, end):
                new_start = cen - win_size // 2
                new_end = new_start + win_size
                new_start = max(0, new_start)
                new_end = min(vlen, new_end)
                new_item = {'video_file': vfile,
                            'name': '{}_{}_[{}:{}]'.format(item['label'], vfile, new_start, new_end),
                            'label': item['label'],
                            'seq_len': new_end - new_start,
                            'start': new_start,
                            'end': new_end,
                            'bag': bag_idx,
                            'aug': 1,
                            'base_start': base_start,
                            'base_end': base_end}
                bag2items[str(bag_idx)].append(new_item)
            bag_idx += 1

    return bag2items


def build_vocab(bag2items):
    vocab = ['<blank>']
    for k, v in bag2items.items():
        for item in v:
            if item['label'] not in vocab:
                vocab.append(item['label'])
    return vocab


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Sign Augmentation")
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--datasetname', default='phoenix', type=str)
    args = parser.parse_args()

    split = args.split
    dataset_name = args.datasetname

    paths = get_dataset_paths(dataset_name, split)
    # 加载数据
    try:
        data = load_pickle(paths['iso'])
        data1 = load_pickle(paths['iso_blank'])
        ori = load_pickle(paths['ori'], gz=True)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    data = [*data, *data1]

    bag2items = build_bag2items(data, ori)

    # 保存bag2items
    save_pickle(bag2items, paths['bag2items'])

    if split == 'train':
        vocab = build_vocab(bag2items)

        with open(paths['vocab'], 'w') as f:
            json.dump(vocab, f)

    if split in ['dev', 'test']:
        new_data = [item for items in bag2items.values() for item in items]
        save_pickle(new_data, paths['center_label'])
