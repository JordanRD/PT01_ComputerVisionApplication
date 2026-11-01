import argparse
import csv
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

LOG_COLUMNS = [
    "event_id",
    "track_id",
    "class_name",
    "first_seen_frame",
    "last_seen_frame",
    "lost_frame",
    "lost_time_sec",
    "last_x1",
    "last_y1",
    "last_x2",
    "last_y2",
    "last_conf",
    "lost_reason",
    "status",
    "recovered_frame",
    "recovered_as_id",
    "gap_frames",
]


@dataclass
class TrackState:
    class_name: str
    first_seen: int
    last_seen: int
    first_box: tuple
    last_box: tuple
    last_conf: float


@dataclass
class LostEvent:
    event_id: int
    track_id: int
    state: TrackState
    lost_frame: int
    lost_reason: str
    recovered_frame: int | None = None
    recovered_as_id: int | None = None


def box_iou(a: tuple, b: tuple) -> float:
    inter_w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class LostObjectDetector:
    """Membandingkan ID pada tiap frame dengan riwayat ID sebelumnya.

    Sebuah ID dianggap hilang jika absen selama `patience` frame berturut-turut.
    Dengan patience=1 logikanya murni perbandingan frame t-1 dan frame t.

    Tracker bisa memberi ID baru pada obyek diam yang sama setelah track-nya kedaluwarsa.
    Karena kamera statis, ID yang baru muncul dengan kelas sama dan IoU >= `reid_iou`
    terhadap posisi terakhir obyek diam yang hilang dihitung sebagai obyek lama yang
    kembali. Obyek yang bergerak tidak digabung karena posisinya tidak bisa jadi bukti.
    """

    def __init__(self, content_area: tuple, patience: int, edge_margin: int, reid_iou: float):
        self.content_area = content_area
        self.patience = patience
        self.edge_margin = edge_margin
        self.reid_iou = reid_iou
        self.tracks: dict[int, TrackState] = {}
        self.open_events: dict[int, LostEvent] = {}
        self.events: list[LostEvent] = []
        # ID lama yang sudah digantikan ID baru, tidak dipantau lagi
        self.merged_ids: set[int] = set()

    def _lost_reason(self, box: tuple) -> str:
        x1, y1, x2, y2 = box
        left, top, right, bottom = self.content_area
        near_edge = (
            x1 <= left + self.edge_margin
            or y1 <= top + self.edge_margin
            or x2 >= right - self.edge_margin
            or y2 >= bottom - self.edge_margin
        )
        return "keluar_frame" if near_edge else "hilang_di_dalam_frame"

    @property
    def unique_objects(self) -> int:
        return len(self.tracks) - len(self.merged_ids)

    def _reassociate(self, frame_idx: int, detections: dict[int, tuple]):
        for track_id, event in list(self.open_events.items()):
            state = event.state
            # Track pendek belum cukup jadi bukti bahwa obyeknya memang diam
            is_static = (
                state.last_seen - state.first_seen >= self.patience
                and box_iou(state.first_box, state.last_box) >= self.reid_iou
            )
            if not is_static:
                continue

            for other_id, (class_name, box, _) in detections.items():
                other = self.tracks[other_id]
                if (
                    class_name == state.class_name
                    and other.first_seen > state.last_seen
                    and frame_idx - other.first_seen <= self.patience
                    and box_iou(box, state.last_box) >= self.reid_iou
                ):
                    event.recovered_frame = other.first_seen
                    event.recovered_as_id = other_id
                    del self.open_events[track_id]
                    self.merged_ids.add(track_id)
                    break

    def update(self, frame_idx: int, detections: dict[int, tuple]) -> list[LostEvent]:
        """detections: {track_id: (class_name, box, conf)}. Mengembalikan event baru."""
        for track_id, (class_name, box, conf) in detections.items():
            self.merged_ids.discard(track_id)
            state = self.tracks.get(track_id)
            if state is None:
                self.tracks[track_id] = TrackState(class_name, frame_idx, frame_idx, box, box, conf)
            else:
                state.last_seen = frame_idx
                state.last_box = box
                state.last_conf = conf

            # ID yang sudah ditandai hilang muncul lagi di frame ini
            event = self.open_events.pop(track_id, None)
            if event is not None:
                event.recovered_frame = frame_idx

        new_events = []
        for track_id, state in self.tracks.items():
            if track_id in detections or track_id in self.open_events or track_id in self.merged_ids:
                continue
            if frame_idx - state.last_seen >= self.patience:
                # Salin state supaya posisi terakhir pada event tidak berubah saat ID muncul lagi
                snapshot = TrackState(**vars(state))
                event = LostEvent(
                    event_id=len(self.events) + 1,
                    track_id=track_id,
                    state=snapshot,
                    lost_frame=state.last_seen + 1,
                    lost_reason=self._lost_reason(state.last_box),
                )
                self.open_events[track_id] = event
                self.events.append(event)
                new_events.append(event)

        self._reassociate(frame_idx, detections)
        return new_events


def parse_detections(result) -> dict[int, tuple]:
    boxes = result.boxes
    if boxes is None or boxes.id is None:
        return {}

    detections = {}
    for track_id, cls, box, conf in zip(
        boxes.id.int().tolist(), boxes.cls.int().tolist(), boxes.xyxy.tolist(), boxes.conf.tolist()
    ):
        detections[track_id] = (result.names[cls], tuple(int(v) for v in box), round(conf, 3))
    return detections


def find_content_area(video_path: str, sample_every: int = 20, threshold: int = 20) -> tuple:
    """Video CCTV bisa punya bar hitam, jadi tepi gambar belum tentu tepi frame."""
    cap = cv2.VideoCapture(video_path)
    brightest = None
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightest = gray if brightest is None else cv2.max(brightest, gray)
        frame_idx += 1
    cap.release()

    rows = np.where(brightest.mean(axis=1) > threshold)[0]
    cols = np.where(brightest.mean(axis=0) > threshold)[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def draw_dashed_rect(frame, box: tuple, color: tuple, thickness: int = 2, dash: int = 12):
    x1, y1, x2, y2 = box
    for x in range(x1, x2, dash * 2):
        cv2.line(frame, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(frame, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(frame, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(frame, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def draw_lost_markers(frame, detector: LostObjectDetector, frame_idx: int, marker_frames: int, visible: int):
    red = (0, 0, 255)
    frame_width = frame.shape[1]
    for event in detector.open_events.values():
        # Event baru tercatat setelah `patience` frame, jadi durasi marker dihitung dari situ
        if frame_idx - event.lost_frame - detector.patience > marker_frames:
            continue
        x1, y1, x2, y2 = event.state.last_box
        draw_dashed_rect(frame, event.state.last_box, red, thickness=1, dash=4)
        label = f"HILANG {event.track_id} {event.state.class_name}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
        text_x = max(0, min(x1, frame_width - text_w - 4))
        text_y = max(y1, text_h + 4)
        cv2.rectangle(frame, (text_x, text_y - text_h - 4), (text_x + text_w + 4, text_y), red, -1)
        cv2.putText(frame, label, (text_x + 2, text_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    summary = f"Terlihat: {visible} | Hilang kumulatif: {len(detector.open_events)} | Obyek unik: {detector.unique_objects}"
    cv2.putText(frame, summary, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)


def write_lost_log(path: str, events: list[LostEvent], fps: float):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LOG_COLUMNS)
        for event in events:
            state = event.state
            recovered = event.recovered_frame
            writer.writerow([
                event.event_id,
                event.track_id,
                state.class_name,
                state.first_seen,
                state.last_seen,
                event.lost_frame,
                round(event.lost_frame / fps, 2),
                *state.last_box,
                state.last_conf,
                event.lost_reason,
                "lost" if recovered is None else "recovered",
                "" if recovered is None else recovered,
                "" if event.recovered_as_id is None else event.recovered_as_id,
                "" if recovered is None else recovered - event.lost_frame,
            ])


def run_lost_detection(
    video_path: str = "videos/Normal_Videos_606_x264.mp4",
    weights_path: str = "weights/yolov8n.pt",
    conf: float = 0.4,
    tracker: str = "bytetrack.yaml",
    patience: int = 15,
    edge_margin: int = 15,
    reid_iou: float = 0.5,
    marker_seconds: float = 2.0,
    output_dir: str = "runs/lost_objects",
):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    video_output = os.path.join(output_dir, "lost_objects_output.mp4")
    log_output = os.path.join(output_dir, "lost_objects_log.csv")
    per_frame_output = os.path.join(output_dir, "lost_objects_per_frame.csv")

    out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    content_area = find_content_area(video_path)
    detector = LostObjectDetector(content_area, patience, edge_margin, reid_iou)
    marker_frames = int(marker_seconds * fps)

    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(weights_path)

    print(f"[INFO] Tracking video: {video_path}")
    print(f"[INFO] Configuration: conf={conf}, tracker={tracker}, patience={patience}, reid_iou={reid_iou}")
    print(f"[INFO] Content area (left, top, right, bottom): {content_area}")

    results = model.track(
        source=video_path,
        conf=conf,
        persist=True,
        tracker=tracker,
        verbose=False,
        stream=True,
    )

    frame_idx = -1
    with open(per_frame_output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "total_visible", "visible_ids", "total_lost", "lost_ids", "new_lost_ids"])

        for frame_idx, result in enumerate(results):
            detections = parse_detections(result)
            new_events = detector.update(frame_idx, detections)

            writer.writerow([
                frame_idx,
                len(detections),
                " ".join(str(i) for i in sorted(detections)),
                len(detector.open_events),
                " ".join(str(i) for i in sorted(detector.open_events)),
                " ".join(str(e.track_id) for e in new_events),
            ])

            annotated_frame = result.plot(line_width=1)
            draw_lost_markers(annotated_frame, detector, frame_idx, marker_frames, len(detections))
            out.write(annotated_frame)

            if (frame_idx + 1) % 50 == 0:
                print(f"[INFO] Processed {frame_idx + 1} frames...", end="\r")

    out.release()
    write_lost_log(log_output, detector.events, fps)

    recovered = sum(e.recovered_frame is not None for e in detector.events)
    print(f"\n[SUCCESS] Completed processing {frame_idx + 1} frames.")
    print(f"[SUCCESS] Tracker IDs: {len(detector.tracks)}, unique objects: {detector.unique_objects}")
    print(f"[SUCCESS] Lost events: {len(detector.events)}, recovered: {recovered}")
    print(f"[SUCCESS] Log saved to: {log_output}")
    print(f"[SUCCESS] Per-frame data saved to: {per_frame_output}")
    print(f"[SUCCESS] Video saved to: {video_output}")


def main():
    parser = argparse.ArgumentParser(description="Deteksi objek hilang berbasis ID tracking")
    parser.add_argument("--source", type=str, default="videos/Normal_Videos_606_x264.mp4", help="Path to input video")
    parser.add_argument("--weights", type=str, default="weights/yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config (bytetrack.yaml or botsort.yaml)")
    parser.add_argument("--patience", type=int, default=15, help="Frames an ID must be absent before it counts as lost")
    parser.add_argument("--edge-margin", type=int, default=15, help="Pixels from the border that count as leaving the frame")
    parser.add_argument("--reid-iou", type=float, default=0.5, help="IoU for matching a new ID to a lost object")
    parser.add_argument("--marker-seconds", type=float, default=2.0, help="How long the lost marker stays on the video")
    parser.add_argument("--output-dir", type=str, default="runs/lost_objects", help="Output directory")
    args = parser.parse_args()

    run_lost_detection(
        video_path=args.source,
        weights_path=args.weights,
        conf=args.conf,
        tracker=args.tracker,
        patience=args.patience,
        edge_margin=args.edge_margin,
        reid_iou=args.reid_iou,
        marker_seconds=args.marker_seconds,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
