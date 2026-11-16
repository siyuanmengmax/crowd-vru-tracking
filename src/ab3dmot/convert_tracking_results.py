#!/usr/bin/env python3
"""
Convert AB3DMOT tracking results back to the roadside format for evaluation

AB3DMOT output format (KITTI MOT):
frame id type truncated occluded alpha x1 y1 x2 y2 h w l x y z rot_y score

converted to (same columns as the detection format):
frame_id,track_id,label,confidence,x,y,z,w,l,h,yaw

so that it can be evaluated directly against the ground truth
"""

import numpy as np
import pandas as pd
import os
import argparse
import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Convert AB3DMOT results to custom format')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='AB3DMOT result directory (e.g., results/ROADSIDE/xxx/data_0/)')
    parser.add_argument('--output_file', type=str, default='tracking_results.csv',
                        help='Output CSV file')
    parser.add_argument('--reverse_label_map', type=str, default='1:0,0:1,3:3',
                        help='Reverse label mapping: KITTI->Your labels (e.g., "1:0,0:1")')
    parser.add_argument('--coord_transform', type=str, default='roadside',
                        choices=['roadside', 'none'],
                        help='Coordinate transformation mode')
    return parser.parse_args()


def pseudo_camera_to_roadside(x_cam, y_cam, z_cam, h_cam, w_cam, l_cam, rot_y):
    """
    Pseudo-camera coordinates back to roadside LiDAR coordinates

    Inverse of roadside_to_pseudo_camera() in convert_roadside_lidar.py
    """
    # inverse mapping
    x = x_cam
    y = z_cam
    z = -y_cam  # y_cam is negated, undo it

    # sizes unchanged
    w = w_cam
    l = l_cam
    h = h_cam

    # angle unchanged
    yaw = rot_y

    return x, y, z, w, l, h, yaw


def read_kitti_tracking_file(filepath):
    """
    Read tracking results in KITTI MOT format

    format: frame id type truncated occluded alpha x1 y1 x2 y2 h w l x y z rot_y score
    """
    try:
        data = pd.read_csv(filepath, sep=' ', header=None)
        data.columns = ['frame', 'id', 'type', 'truncated', 'occluded', 'alpha',
                       'x1', 'y1', 'x2', 'y2', 'h', 'w', 'l', 'x', 'y', 'z', 'rot_y', 'score']
        return data
    except pd.errors.EmptyDataError:
        # empty file -> empty DataFrame
        return pd.DataFrame()


def convert_results(input_dir, output_file, reverse_label_map, coord_transform):
    """
    Convert AB3DMOT results to the roadside format
    """
    print("="*70)
    print("AB3DMOT tracking result conversion")
    print("="*70)

    # parse the reverse label map
    label_mapping = {}
    label_names = {0: 'person', 1: 'small vehicle', 2: 'large vehicle', 3: 'other'}

    for pair in reverse_label_map.split(','):
        src, dst = pair.split(':')
        label_mapping[src] = int(dst)  # KITTI label name -> your label id

    print(f"\nKITTI class -> roadside class:")
    kitti_names = {'Pedestrian': 1, 'Car': 0, 'Cyclist': 3}
    for kitti_name, kitti_id in kitti_names.items():
        kitti_str = str(kitti_id)
        if kitti_str in label_mapping:
            your_id = label_mapping[kitti_str]
            your_name = label_names.get(your_id, f'class_{your_id}')
            print(f"  {kitti_name} ({kitti_id}) -> {your_name} ({your_id})")

    # find all result files
    print(f"\ninput directory: {input_dir}")
    result_files = glob.glob(os.path.join(input_dir, '*.txt'))

    if len(result_files) == 0:
        print(f"Error: no tracking result files in {input_dir}")
        return

    print(f"tracking results for {len(result_files)} sequences")

    # process all result files
    all_tracks = []

    for result_file in sorted(result_files):
        seq_name = os.path.basename(result_file).replace('.txt', '')
        print(f"\nsequence: {seq_name}")

        # read tracking results
        tracks_df = read_kitti_tracking_file(result_file)

        if len(tracks_df) == 0:
            print(f"  sequence {seq_name} has no tracking results, skipped")
            continue

        print(f"  tracks: {tracks_df['id'].nunique()}")
        print(f"  frames: {tracks_df['frame'].nunique()}")
        print(f"  boxes: {len(tracks_df)}")

        # convert every track row
        for idx, row in tracks_df.iterrows():
            frame_id = int(row['frame'])
            track_id = int(row['id'])

            # class mapping
            kitti_type = row['type']
            kitti_type_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 3}.get(kitti_type, -1)
            kitti_type_str = str(kitti_type_id)

            if kitti_type_str not in label_mapping:
                print(f"  warning: unknown class {kitti_type} (ID={kitti_type_id}), skipped")
                continue

            label = label_mapping[kitti_type_str]
            confidence = float(row['score'])

            # 3D box parameters (camera coordinates)
            x_cam = float(row['x'])
            y_cam = float(row['y'])
            z_cam = float(row['z'])
            h_cam = float(row['h'])
            w_cam = float(row['w'])
            l_cam = float(row['l'])
            rot_y = float(row['rot_y'])

            # coordinate conversion
            if coord_transform == 'roadside':
                # back to roadside coordinates
                x, y, z, w, l, h, yaw = pseudo_camera_to_roadside(
                    x_cam, y_cam, z_cam, h_cam, w_cam, l_cam, rot_y
                )
            else:
                # unchanged
                x, y, z = x_cam, y_cam, z_cam
                w, l, h = w_cam, l_cam, h_cam
                yaw = rot_y

            # append to the result list
            all_tracks.append({
                'frame_id': frame_id,
                'track_id': track_id,
                'label': label,
                'confidence': confidence,
                'x': x,
                'y': y,
                'z': z,
                'w': w,
                'l': l,
                'h': h,
                'yaw': yaw
            })

    # to DataFrame and save
    if len(all_tracks) == 0:
        print("\nError: no valid tracking results")
        return

    results_df = pd.DataFrame(all_tracks)

    # sort by frame_id
    results_df = results_df.sort_values(['frame_id', 'track_id'])

    # save
    results_df.to_csv(output_file, index=False)

    print("\n" + "="*70)
    print("Conversion finished.")
    print("="*70)

    # statistics
    print(f"\noutput file: {output_file}")
    print(f"track rows: {len(all_tracks)}")
    print(f"unique IDs: {results_df['track_id'].nunique()}")
    print(f"frame range: {results_df['frame_id'].min()} - {results_df['frame_id'].max()}")

    print("\nclass distribution:")
    label_counts = results_df['label'].value_counts()
    for label_id, count in label_counts.items():
        label_name = {0: 'person', 1: 'small vehicle', 2: 'large vehicle', 3: 'other'}.get(label_id, f'class_{label_id}')
        print(f"  {label_name} ({label_id}): {count}")

    print("\npreview:")
    print(results_df.head(10))

    print("\nnext: evaluate this file against the ground truth")


def main():
    args = parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: input directory not found: {args.input_dir}")
        return

    convert_results(
        input_dir=args.input_dir,
        output_file=args.output_file,
        reverse_label_map=args.reverse_label_map,
        coord_transform=args.coord_transform
    )


if __name__ == '__main__':
    main()
