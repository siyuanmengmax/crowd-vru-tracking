#!/usr/bin/env python3
"""
Single-point tracker: UKF + Hungarian assignment.
Tracks only the box centre (x, y, z); box dimensions are not used for association.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints


class SimpleUKFTracker:
    """Simple UKF track: 3D position only."""

    def __init__(self, initial_position, tracker_id):
        """
        Initialise the UKF track.

        Args:
            initial_position: [x, y, z] initial position
            tracker_id: track ID
        """
        self.tracker_id = tracker_id
        self.dim_x = 6  # state: [x, y, z, vx, vy, vz]
        self.dim_z = 3  # measurement: [x, y, z]

        # sigma points
        points = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.1, beta=2., kappa=1.)

        # UKF
        self.ukf = UKF(dim_x=self.dim_x, dim_z=self.dim_z, dt=0.1,
                       hx=self.measurement_function, fx=self.state_transition_function,
                       points=points)

        # initial state: [x, y, z, vx, vy, vz]
        self.ukf.x = np.array([
            initial_position[0], initial_position[1], initial_position[2],
            0.0, 0.0, 0.0  # zero initial velocity
        ])

        # initial state covariance
        self.ukf.P = np.eye(self.dim_x) * 10.0
        self.ukf.P[3:, 3:] *= 5.0  # larger uncertainty for velocity

        # process noise covariance
        self.ukf.Q = np.eye(self.dim_x)
        self.ukf.Q[:3, :3] *= 0.1  # position process noise
        self.ukf.Q[3:, 3:] *= 1.0  # velocity process noise

        # measurement noise covariance
        self.ukf.R = np.eye(self.dim_z) * 0.5

        # track history
        self.history = []
        self.age = 0
        self.hits = 0
        self.time_since_update = 0

    @staticmethod
    def state_transition_function(x, dt):
        """State transition: constant-velocity model."""
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        return F @ x

    @staticmethod
    def measurement_function(x):
        """Measurement function: position only."""
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        return H @ x

    def predict(self):
        """Predict the next state."""
        self.ukf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, measurement):
        """
        Update the track.

        Args:
            measurement: [x, y, z] measurement
        """
        self.ukf.update(np.array(measurement))
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.get_state())

    def get_state(self):
        """Current state (position)."""
        return self.ukf.x[:3].copy()

    def get_velocity(self):
        """Current velocity."""
        return self.ukf.x[3:].copy()


class HungarianMatcher:
    """Data association with the Hungarian algorithm."""

    def __init__(self, max_distance=5.0):
        """
        Initialise the matcher.

        Args:
            max_distance: maximum association distance (m)
        """
        self.max_distance = max_distance

    def associate(self, detections, trackers):
        """
        Associate detections with tracks using the Hungarian algorithm.

        Args:
            detections: list of detections, each [x, y, z, ...]
            trackers: list of tracks

        Returns:
            matches: list of matched pairs [(det_idx, trk_idx), ...]
            unmatched_detections: indices of unmatched detections
            unmatched_trackers: indices of unmatched tracks
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []

        if len(detections) == 0:
            return [], [], list(range(len(trackers)))

        # cost matrix (Euclidean distance)
        cost_matrix = np.zeros((len(detections), len(trackers)))

        for d, det in enumerate(detections):
            det_pos = np.array([det[0], det[1], det[2]])
            for t, tracker in enumerate(trackers):
                trk_pos = tracker.get_state()
                cost_matrix[d, t] = np.linalg.norm(det_pos - trk_pos)

        # optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # discard matches beyond the distance threshold
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(trackers)))

        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] <= self.max_distance:
                matches.append((row, col))
                unmatched_detections.remove(row)
                unmatched_trackers.remove(col)

        return matches, unmatched_detections, unmatched_trackers


class SinglePointTracker:
    """Single-point tracking system."""

    def __init__(self, csv_path, max_distance=5.0, max_age=5):
        """
        Initialise the tracking system.

        Args:
            csv_path: path of detection_bbox_results.csv
            max_distance: maximum association distance of the Hungarian matching (m)
            max_age: maximum number of frames a track survives without an update
        """
        self.csv_path = Path(csv_path)
        self.max_distance = max_distance
        self.max_age = max_age

        # read data
        print(f"Reading data: {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"{len(self.df)} detections read")
        print(f"Frame range: {self.df['frame_id'].min()} - {self.df['frame_id'].max()}")

        # tracks
        self.trackers = []
        self.next_tracker_id = 0

        # Hungarian matcher
        self.matcher = HungarianMatcher(max_distance=max_distance)

        # tracking results
        self.tracking_results = []

    def run(self):
        """Run tracking."""
        print("\nStarting single-point tracking...")

        # all frames
        frames = sorted(self.df['frame_id'].unique())

        for frame_id in tqdm(frames, desc="Tracking"):
            # detections of the current frame
            frame_detections = self.df[self.df['frame_id'] == frame_id]

            # list format: [x, y, z, cluster_id, frame_id, confidence, label, w, l, h, yaw]
            detections = []
            for _, row in frame_detections.iterrows():
                detection = [
                    row['x'], row['y'], row['z'],
                    row['cluster_id'], row['frame_id'],
                    row.get('confidence', 1.0),
                    row.get('label', -1),
                    row.get('w', 1.0),
                    row.get('l', 1.0),
                    row.get('h', 1.0),
                    row.get('yaw', 0.0)
                ]
                detections.append(detection)

            # predict all tracks
            for tracker in self.trackers:
                tracker.predict()

            # map from detection index to track
            det_to_tracker = {}

            # data association
            if len(detections) > 0 and len(self.trackers) > 0:
                matches, unmatched_dets, unmatched_trks = self.matcher.associate(
                    detections, self.trackers
                )

                # update matched tracks and record the mapping
                for det_idx, trk_idx in matches:
                    measurement = [detections[det_idx][0],
                                 detections[det_idx][1],
                                 detections[det_idx][2]]
                    self.trackers[trk_idx].update(measurement)
                    # record detection -> track
                    det_to_tracker[det_idx] = self.trackers[trk_idx]

                # remove unmatched tracks that are too old
                trackers_to_remove = []
                for trk_idx in unmatched_trks:
                    if self.trackers[trk_idx].time_since_update > self.max_age:
                        trackers_to_remove.append(trk_idx)

                # delete from the back to keep indices valid
                for trk_idx in sorted(trackers_to_remove, reverse=True):
                    del self.trackers[trk_idx]

                # new tracks for unmatched detections, and record the mapping
                for det_idx in unmatched_dets:
                    initial_pos = [detections[det_idx][0],
                                 detections[det_idx][1],
                                 detections[det_idx][2]]
                    new_tracker = SimpleUKFTracker(initial_pos, self.next_tracker_id)
                    new_tracker.update(initial_pos)  # update once immediately
                    self.trackers.append(new_tracker)
                    # record detection -> new track
                    det_to_tracker[det_idx] = new_tracker
                    self.next_tracker_id += 1

            elif len(detections) > 0:
                # no tracks yet: a new track for every detection
                for i, det in enumerate(detections):
                    initial_pos = [det[0], det[1], det[2]]
                    new_tracker = SimpleUKFTracker(initial_pos, self.next_tracker_id)
                    new_tracker.update(initial_pos)
                    self.trackers.append(new_tracker)
                    # record detection -> new track
                    det_to_tracker[i] = new_tracker
                    self.next_tracker_id += 1

            # store the results of the current frame
            for i, det in enumerate(detections):
                # look up the track of this detection
                if i in det_to_tracker:
                    tracker = det_to_tracker[i]
                    matched_tracker_id = tracker.tracker_id
                    tracker_state = tracker.get_state()
                    tracker_velocity = tracker.get_velocity()
                else:
                    # should not happen
                    matched_tracker_id = -1
                    tracker_state = [det[0], det[1], det[2]]
                    tracker_velocity = [0.0, 0.0, 0.0]

                result = {
                    'frame_id': frame_id,
                    'track_id': matched_tracker_id,
                    'label': det[6] if len(det) > 6 else -1,
                    'confidence': det[5] if len(det) > 5 else 1.0,
                    'x': det[0],
                    'y': det[1],
                    'z': det[2],
                    'w': det[7] if len(det) > 7 else 1.0,
                    'l': det[8] if len(det) > 8 else 1.0,
                    'h': det[9] if len(det) > 9 else 1.0,
                    'yaw': det[10] if len(det) > 10 else 0.0
                }
                self.tracking_results.append(result)

        print(f"\nTracking finished.")
        print(f"Tracking records: {len(self.tracking_results)}")
        print(f"Tracks created: {self.next_tracker_id}")

        return self.tracking_results

    def postprocess_labels(self):
        """
        Post-process the label field per track_id with a priority rule:
        1. if any label in the track is 2 (large vehicle), set all to 2;
        2. otherwise, if any label is 1 (small vehicle), set all to 1;
        3. otherwise, use the most frequent label of the track.
        """
        if not self.tracking_results:
            return

        print("\nStarting label post-processing...")

        # DataFrame for convenience
        df = pd.DataFrame(self.tracking_results)

        # per track_id
        for track_id in df['track_id'].unique():
            if track_id < 0:  # skip invalid track_id
                continue

            track_mask = df['track_id'] == track_id
            track_labels = df.loc[track_mask, 'label'].values

            # priority rule
            if 2 in track_labels:
                # rule 1: large vehicle
                final_label = 2
            elif 1 in track_labels:
                # rule 2: small vehicle
                final_label = 1
            else:
                # rule 3: most frequent label
                from collections import Counter
                label_counts = Counter(track_labels)
                final_label = label_counts.most_common(1)[0][0]

            # set the label of the whole track
            df.loc[track_mask, 'label'] = final_label

        # write back to tracking_results
        self.tracking_results = df.to_dict('records')

        # statistics
        label_dist = df.groupby('label').size()
        print(f"Label post-processing finished.")
        print(f"Label distribution:")
        for label, count in label_dist.items():
            label_name = {0: "pedestrian", 1: "small vehicle", 2: "large vehicle", 3: "other"}.get(label, f"unknown({label})")
            print(f"  - {label_name}: {count} records")

    def save_results(self, output_path="tracking_single_point_results.csv"):
        """Save the tracking results (same columns as benchmark_results.csv)."""
        if self.tracking_results:
            df = pd.DataFrame(self.tracking_results)

            # column order: frame_id, track_id, label, confidence, x, y, z, w, l, h, yaw
            column_order = ['frame_id', 'track_id', 'label', 'confidence', 'x', 'y', 'z', 'w', 'l', 'h', 'yaw']
            df = df[column_order]

            df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")

            # statistics
            total_tracks = df['track_id'].nunique()
            avg_track_length = df.groupby('track_id').size().mean()

            print(f"\nTracking statistics:")
            print(f"  - tracks: {total_tracks}")
            print(f"  - mean track length: {avg_track_length:.1f} frames")
            print(f"  - tracked points: {len(df)}")

            # label distribution
            if 'label' in df.columns:
                print(f"\n  - Label distribution:")
                for label in sorted(df['label'].unique()):
                    count = (df['label'] == label).sum()
                    label_name = {0: "pedestrian", 1: "small vehicle", 2: "large vehicle", 3: "other"}.get(label, f"unknown({label})")
                    print(f"    {label_name}: {count}")

            return df
        else:
            print("No tracking results to save")
            return None


def main():
    """Command-line entry point."""
    print("=" * 60)
    print("Single-point tracker - UKF + Hungarian algorithm")
    print("=" * 60)

    csv_path = "detection_bbox_results.csv"

    if not Path(csv_path).exists():
        print(f"Error: file not found {csv_path}")
        return

    # tracker
    tracker = SinglePointTracker(
        csv_path=csv_path,
        max_distance=3,  # maximum association distance (m)
        max_age=3  # maximum frames without update
    )

    # run tracking
    results = tracker.run()

    # label post-processing
    if results:
        tracker.postprocess_labels()

    # save results
    if results:
        df = tracker.save_results()

        print("\nSingle-point tracking finished.")
        print("\nComponents:")
        print("  - UKF (unscented Kalman filter), 6D state [x,y,z,vx,vy,vz]")
        print("  - Hungarian algorithm, Euclidean-distance association")
        print("  - box centre only")
        print("  - label post-processing, priority rule (large vehicle > small vehicle > other)")
        print("  - output columns as benchmark_results.csv")


if __name__ == "__main__":
    main()
