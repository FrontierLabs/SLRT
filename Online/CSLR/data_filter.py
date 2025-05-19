#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time: 2025/5/16 14:18
# @Author: adacyang
# @FileName: data_filter.py
import argparse
import pickle
import random

from typing import Dict, List, Any


def check_black(segment) -> float:
    """
    计算segment中label为'<blank>'的比例。

    Args:
        segment: 字典，值为包含'label'键的字典列表。

    Returns:
        float: '<blank>'标签的比例。如果没有标签，返回0.0。
    """
    total = 0
    blank_count = 0

    if isinstance(segment, Dict):
        for values in segment.values():
            for item in values:
                label = item.get('label')
                total += 1
                if label == '<blank>':
                    blank_count += 1
    else:
        for item in segment:
            label = item.get('label')
            total += 1
            if label == '<blank>':
                blank_count += 1

    return blank_count / total if total > 0 else 0.0


def blank_filter(
    segment,
    output_file: str,
    blank_keep_prob: float = 0.1
) -> None:
    """
    过滤segment中的数据，去除大部分label为<blank>的项（保留概率为blank_keep_prob），
    如果某个key下有label为<unk>的项，则丢弃该key下所有项。
    结果以pickle格式保存到output_file。

    Args:
        segment: 输入数据，key为任意类型，value为dict的list，dict需有'label'键。
        output_file: 输出文件路径。
        blank_keep_prob: 保留<blank>项的概率，默认0.1（10%）。
    """
    if isinstance(segment, Dict):
        filtered_segment = dict()

        for key, values in segment.items():
            # 如果有<unk>，直接丢弃
            if any(item['label'] == '<unk>' for item in values):
                continue

            # 过滤<blank>
            new_values = [
                item for item in values
                if item['label'] != '<blank>' or random.random() < blank_keep_prob
            ]

            if new_values:
                filtered_segment[key] = new_values
    else:
        filtered_segment = []

        for item in segment:
            if item['label'] == '<unk>':
                continue

            if item['label'] != '<blank>' or random.random() < blank_keep_prob:
                filtered_segment.append(item)

    # 保存
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(filtered_segment, f)
    except Exception as e:
        print(f"Error saving file {output_file}: {e}")

    blank_rate = check_black(filtered_segment)
    print(f'{output_file} blank_rate: {blank_rate}')


def filter_videos_by_unk_ratio(
    segments: List[Dict[str, Any]],
    data: List[Dict[str, Any]],
    output_file: str,
    unk_label: str = '<unk>',
    threshold: float = 0.05
) -> None:
    """
    过滤gloss序列中，unk比例大于阈值的视频，并保存过滤后的数据。

    Args:
        segments: 包含视频分段和标签的字典列表，每个元素需有'video_file'和'label'键。
        data: 原始数据列表，每个元素需有'name'键。
        output_file: 过滤后数据的保存路径。
        unk_label: 视为未知的标签，默认'<unk>'。
        threshold: 过滤阈值，默认0.05。
    """
    # 统计每个视频的分段
    video_to_segments: Dict[str, List[Dict[str, Any]]] = {}
    for item in segments:
        video_file = item['video_file']
        video_to_segments.setdefault(video_file, []).append(item)

    print(f'视频总量：{len(video_to_segments)}')

    # 过滤unk比例大于阈值的视频
    keep_videos = []
    filtered_videos = []
    for video_file, segs in video_to_segments.items():
        unk_count = sum(1 for seg in segs if seg['label'] == unk_label)
        unk_ratio = unk_count / len(segs)
        if unk_ratio > threshold:
            filtered_videos.append(video_file)
        else:
            keep_videos.append(video_file)

    print(f'过滤gloss序列中，unk比例大于{threshold}的视频，过滤数量为：{len(filtered_videos)}')

    # 保留未被过滤的视频数据
    filtered_data = [item for item in data if item['name'] in keep_videos]

    # 保存结果
    try:
        with open(output_file, 'wb') as f:
            pickle.dump(filtered_data, f)
        print(f'过滤后数据已保存到: {output_file}')
    except Exception as e:
        print(f'保存文件时出错: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DataFilter")
    parser.add_argument('--split', default='test', type=str,
                        choices=['test', 'dev', 'train'],
                        help='Specify the dataset split to use. Options are: test, dev, train.')
    parser.add_argument('--filter_func', default='blank', type=str,
                        choices=['blank', 'unk_video'])
    args = parser.parse_args()

    if args.split == 'train':
        segment_file = '../../data/ce_csl/ce_csl_iso_center_label_bag2items.train'
    else:
        segment_file = f'../../data/ce_csl/ce_csl_iso_center_label.{args.split}'

    with open(segment_file, 'rb') as f:
        segment = pickle.load(f)

    if args.filter_func == 'blank':
        assert args.split in ['train', 'dev'],\
            'Reduce the proportion of blank segments, only support train and dev sets'
        blank_rate = check_black(segment)
        print(f'{segment_file} blank_rate: {blank_rate}')

        if blank_rate >= 0.5:
            output_file = segment_file.replace(f'.{args.split}', f'_new.{args.split}')
            blank_filter(segment, output_file)
    else:
        assert args.split == 'test', 'Delete videos with high unk ratio, only support test sets'

        data_file = f'../../data/ce_csl/{args.split}.pkl'
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        output_file = f'../../data/ce_csl/{args.split}_new.pkl'

        filter_videos_by_unk_ratio(segment, data, output_file)
