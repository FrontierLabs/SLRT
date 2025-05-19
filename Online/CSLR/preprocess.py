#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/5/16 13:06
# @Author: adacyang
# @FileName: preprocess.py
import argparse
import os
import pickle
import re

import pandas as pd


def slice_list(lst, n):
    """将列表 lst 按固定大小 n 进行切片"""
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def process_string(s):
    # 定义一个正则表达式模式，匹配与中文字符相连的“1”
    pattern = r'(?<=[\u4e00-\u9fff])[1|2|3]'

    # 使用re.sub()函数将匹配到的部分替换为空字符串
    processed_string = re.sub(pattern, '', s)

    return processed_string


def prepare_ce_csl_text_data(
    df: pd.DataFrame,
    split: str,
    video_folder: str,
    output_folder: str,
    cut_n: int = 64,
    dev_split: str = "dev"
) -> None:
    """
    处理ce_csl手语数据，生成训练/验证集/测试集的文本和切片信息。

    Args:
        df: 包含数据的DataFrame
        split: 数据集划分（如'train', 'dev', 'test'）
        video_folder: 视频帧文件夹根目录
        output_folder: 输出文件夹
        cut_n: 每段切片帧数
        dev_split: dev集名称（默认'dev'）
    """

    data_for_kp = []
    data = []

    for _, row in df.iterrows():
        number = row['number']
        translator = row['Translator']
        chinese_sentences = row['Chinese Sentences']
        gloss = row['Gloss']
        note = row['Note']

        folder = os.path.join(video_folder, f'{split}/{translator}/{number}')
        if not os.path.exists(folder):
            continue

        frame_lst = os.listdir(folder)
        frame_lst = sorted(frame_lst, key=lambda x: int(x.split('.')[0]))
        frame_num = len(frame_lst)

        # 处理切片数据，用于批量生成keypoint数据
        cut_frame_lst = slice_list(frame_lst, cut_n)
        for i, cut_lst in enumerate(cut_frame_lst):
            data_for_kp.append({
                'number': number,
                'translator': translator,
                'chinese_sentences': chinese_sentences,
                'gloss': gloss,
                'note': note,
                'name': f'{split}/{translator}/{number}/cut_{i}',
                'seq_len': len(cut_lst),
                'num_frames': frame_num,
                'video_file': f'{split}/{translator}/{number}'
            })

        # 处理文本数据：用于完整数据测试
        gloss = process_string(gloss)
        gloss = gloss.split('/')
        gloss = ' '.join(gloss)

        chinese_sentences = ' '.join(chinese_sentences)

        data.append({
            'number': number,
            'translator': translator,
            'text': chinese_sentences,
            'gloss': gloss,
            'note': note,
            'name': f'{split}/{translator}/{number}',
            'num_frames': frame_num,
            'video_file': f'{split}/{translator}/{number}'
        })

    print(f'data_for_kp: {len(data_for_kp)}; data: {len(data)}')
    pickle.dump(data_for_kp, open(os.path.join(output_folder, f'{split}_for_kp.pkl'), 'wb'))
    pickle.dump(data, open(os.path.join(output_folder, f'{split}.pkl'), 'wb'))


def merge_pkls(path, split, from_ckpt=False):
    final = {}
    for fname in os.listdir(path):
        if split in fname and fname != '{:s}.pkl'.format(split):
            with open(os.path.join(path, fname), 'rb') as f:
                try:
                    data = pickle.load(f)
                except:
                    print(fname)
            final.update(data)

    if not from_ckpt:
        with open(path+'/{:s}.pkl'.format(split), 'wb') as f:
            pickle.dump(final, f)
        print("Merged to {:s}/{:s}.pkl".format(path, split))
    else:
        with open(path+'/{:s}_ckpt.pkl'.format(split), 'wb') as f:
            pickle.dump(final, f)
        print("Merged to {:s}/{:s}_ckpt.pkl".format(path, split))


def merge_keypoint_data(input_folder, output_folder):
    all_data = {}

    for split in ['train', 'dev', 'test']:
        own_data = pickle.load(open(
            os.path.join(input_folder, f'{split}.pkl'),
            'rb'))
        for key, value in own_data.items():
            all_data[key] = {
                'keypoints': value
            }

    pickle.dump(all_data, open(os.path.join(output_folder, 'keypoints_hrnet_dark_coco_wholebody_iso.pkl'), 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("PreProcess")
    parser.add_argument('--split', default='test', type=str,
                        choices=['test', 'dev', 'train'],
                        help='Specify the dataset split to use. Options are: test, dev, train.')
    parser.add_argument('--filter_func', default='prepare_text_data', type=str,
                        choices=['prepare_text_data', 'merge_kp'])
    args = parser.parse_args()

    video_folder = '/group/30106/neilzrzhang/2025H1/CE_CSL/data/'
    output_folder = '../../data/ce_csl'

    df = pd.read_csv(f'../../data/ce_csl/{args.split}.csv', index_col=0)
    df = df.fillna('')
    df['number'] = df.index

    if args.filter_func == 'prepare_text_data':
        prepare_ce_csl_text_data(df, args.split, video_folder, output_folder, cut_n=64)
    else:
        input_folder = '../../data/ce_csl/keypoints_hrnet_dark_coco_wholebody'
        # merge_pkls(input_folder, args.split)

        merge_keypoint_data(input_folder, output_folder)
