#!/usr/bin/env python3
"""
Convert roadside LiDAR detections to the AB3DMOT input format

Roadside setting:
1. the LiDAR is static, no ego-motion
2. usually no camera, or the camera is not the main sensor
3. world coordinates instead of camera coordinates
4. elevated view over a large area

Conversion:
- keep LiDAR coordinates, with a simple axis mapping to the AB3DMOT input format
- no ego-motion compensation
- placeholder 2D boxes (not needed at a roadside site)
"""

import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Convert roadside Lidar detection to AB3DMOT format')
    parser.add_argument('input_file', type=str, help='Input detection CSV file')
    parser.add_argument('--output_dir', type=str, default='./data/ROADSIDE/detection',
                        help='Output directory for AB3DMOT format files')
    parser.add_argument('--det_name', type=str, default='roadside_detector',
                        help='Detector name for output folder')
    parser.add_argument('--split', type=str, default='val',
                        help='Data split (val/test/train)')
    parser.add_argument('--label_map', type=str, default='0:0,1:1,3:3',
                        help='Label mapping (e.g., "0:0,1:1,3:3")')
    parser.add_argument('--coord_mode', type=str, default='roadside',
                        choices=['roadside', 'vehicle'],
                        help='Coordinate system mode: roadside (no transform) or vehicle (KITTI-style)')
    return parser.parse_args()


def roadside_to_pseudo_camera(x, y, z, w, l, h, yaw):
    """
    Map roadside LiDAR coordinates to pseudo-camera coordinates

    Roadside LiDAR axes, typically:
    - X: east or along the road
    - Y: north or across the road
    - Z: height

    Pseudo-camera coordinates (same meaning, axes re-oriented only):
    - x_cam: lateral (LiDAR X or Y)
    - y_cam: height, pointing down (-Z)
    - z_cam: depth (the other LiDAR axis)

    Args:
        x, y, z: LiDAR coordinates (scene frame)
        w, l, h: object size
        yaw: heading

    Returns:
        x_cam, y_cam, z_cam, h_cam, w_cam, l_cam, rot_y, alpha
    """
    # simple mapping: X, Y unchanged, Z negated
    # minimal change that satisfies the AB3DMOT input format
    x_cam = x
    y_cam = -z  # height negated, camera y points down
    z_cam = y

    # sizes unchanged
    h_cam = h
    w_cam = w
    l_cam = l

    # angle unchanged (roadside headings are defined in world coordinates)
    rot_y = yaw

    # alpha (set equal to rot_y at a roadside site)
    # the observer does not move, so the viewing angle is fixed
    alpha = yaw

    return x_cam, y_cam, z_cam, h_cam, w_cam, l_cam, rot_y, alpha


def generate_dummy_2d_bbox(obj_id, img_width=1920, img_height=1080):
    """
    Generate a placeholder 2D box

    A roadside site needs no 2D box; this is a placeholder
    placed by object ID to avoid overlaps

    Args:
        obj_id: object ID (sets the position)
        img_width, img_height: assumed image size

    Returns:
        x1, y1, x2, y2
    """
    # small placeholder box, position from the ID
    box_size = 50
    offset = (obj_id * 60) % (img_width - box_size)

    x1 = offset
    y1 = 50
    x2 = x1 + box_size
    y2 = y1 + box_size

    return x1, y1, x2, y2


def convert_detection_file(input_file, output_dir, det_name, split, label_map, coord_mode):
    """
    Convert a roadside LiDAR detection file to the AB3DMOT format
    """
    print("="*70)
    print("Roadside LiDAR detection conversion")
    print("="*70)

    # read the input CSV
    print(f"\nreading detection file: {input_file}")
    df = pd.read_csv(input_file)

    print(f"\ndataset:")
    print(f"  boxes: {len(df)}")
    print(f"  frame range: {df['frame_id'].min()} - {df['frame_id'].max()}")
    print(f"  frames: {df['frame_id'].nunique()}")
    print(f"  labels: {df['label'].unique()}")
    print(f"  coordinate range:")
    print(f"    X: [{df['x'].min():.2f}, {df['x'].max():.2f}]")
    print(f"    Y: [{df['y'].min():.2f}, {df['y'].max():.2f}]")
    print(f"    Z: [{df['z'].min():.2f}, {df['z'].max():.2f}]")

    # parse the label map
    label_mapping = {}
    for pair in label_map.split(','):
        src, dst = pair.split(':')
        label_mapping[int(src)] = int(dst)
    print(f"\nlabel map: {label_mapping}")

    # KITTI label names
    kitti_label_names = {0: 'Car', 1: 'Pedestrian', 3: 'Cyclist'}

    # group by class
    categories = {}
    for src_label, dst_label in label_mapping.items():
        cat_name = kitti_label_names.get(dst_label, f'Class_{dst_label}')
        categories[cat_name] = dst_label

    # each class
    for cat_name, cat_id in categories.items():
        print(f"\n{'='*70}")
        print(f"class: {cat_name} (KITTI ID={cat_id})")
        print('='*70)

        # detections of this class
        src_labels = [k for k, v in label_mapping.items() if v == cat_id]
        cat_df = df[df['label'].isin(src_labels)].copy()

        if len(cat_df) == 0:
            print(f"  no detections of this class, skipped")
            continue

        print(f"  boxes: {len(cat_df)}")
        print(f"  frames: {cat_df['frame_id'].nunique()}")

        # convert
        output_rows = []
        for idx, row in cat_df.iterrows():
            frame_id = int(row['frame_id'])

            # 3D box parameters
            x, y, z = row['x'], row['y'], row['z']
            w, l, h = row['w'], row['l'], row['h']
            yaw = row['yaw']
            confidence = row['confidence']

            # coordinate conversion
            if coord_mode == 'roadside':
                # roadside mode: keep coordinates, simple mapping
                x_cam, y_cam, z_cam, h_cam, w_cam, l_cam, rot_y, alpha = \
                    roadside_to_pseudo_camera(x, y, z, w, l, h, yaw)
            else:
                # vehicle mode: KITTI-style conversion
                x_cam = -y
                y_cam = -z
                z_cam = x
                h_cam, w_cam, l_cam = h, w, l
                rot_y = -yaw + np.pi/2
                rot_y = np.arctan2(np.sin(rot_y), np.cos(rot_y))
                alpha = rot_y - np.arctan2(x_cam, z_cam)
                alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

            # placeholder 2D box
            x1, y1, x2, y2 = generate_dummy_2d_bbox(idx)

            # AB3DMOT format: frame, type, x1, y1, x2, y2, score, h, w, l, x, y, z, rot_y, alpha
            output_row = [
                frame_id,
                cat_id,
                x1, y1, x2, y2,
                confidence,
                h_cam, w_cam, l_cam,
                x_cam, y_cam, z_cam,
                rot_y,
                alpha
            ]
            output_rows.append(output_row)

        # sort by frame
        output_rows.sort(key=lambda x: x[0])

        # output directory
        output_folder = os.path.join(output_dir, f'{det_name}_{cat_name}_{split}')
        os.makedirs(output_folder, exist_ok=True)

        # save
        seq_name = '0000'  # default sequence name
        output_file = os.path.join(output_folder, f'{seq_name}.txt')

        with open(output_file, 'w') as f:
            for row in output_rows:
                line = ','.join([f'{x:.6f}' if isinstance(x, float) else str(x) for x in row])
                f.write(line + '\n')

        print(f"\n  saved: {output_file}")
        print(f"  rows: {len(output_rows)}")

    print("\n" + "="*70)
    print("Conversion finished.")
    print("="*70)

    # next steps
    print("\nNext:")
    print(f"\n1. roadside config (ego_com: false): src/ab3dmot/ROADSIDE.yml")
    print(f"   (used by src/ab3dmot/run_roadside.py)")

    print(f"\n2. run tracking:")
    print(f"   python3 <repo>/src/ab3dmot/run_roadside.py --dataset ROADSIDE --split {split} --det_name {det_name}   (from the AB3DMOT directory)")

    print(f"\n3. results:")
    print(f"   ./results/ROADSIDE/{det_name}_{split}_H1/")

    print("\nNote:")
    print("   a roadside LiDAR has no ego-motion; ego_com must be false")
    print("   (otherwise AB3DMOT fails for lack of oxts data)")


def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: input file not found: {args.input_file}")
        return

    convert_detection_file(
        input_file=args.input_file,
        output_dir=args.output_dir,
        det_name=args.det_name,
        split=args.split,
        label_map=args.label_map,
        coord_mode=args.coord_mode
    )


if __name__ == '__main__':
    main()
