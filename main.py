import argparse
import os
import sys
from ultralytics import YOLO


def run_tracking(
    video_path: str = "videos/Normal_Videos_606_x264.mp4",
    weights_path: str = "weights/yolov8n.pt",
    conf: float = 0.4,
    tracker: str = "bytetrack.yaml",
    project: str = "runs/track",
    name: str = "hasil_deteksi",
    stream: bool = True,
):
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        sys.exit(1)

    if not os.path.exists(weights_path):
        print(f"[WARNING] Local weights not found at {weights_path}, YOLO will download/load default.")

    print(f"[INFO] Loading model: {weights_path}")
    model = YOLO(weights_path)

    print(f"[INFO] Tracking video: {video_path}")
    print(f"[INFO] Configuration: conf={conf}, tracker={tracker}")

    results = model.track(
        source=video_path,
        conf=conf,
        persist=True,
        tracker=tracker,
        save=True,
        project=project,
        name=name,
        exist_ok=True,
        verbose=False,
        stream=stream,
    )

    frame_count = 0
    for _ in results:
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"[INFO] Processed {frame_count} frames...", end="\r")

    print(f"\n[SUCCESS] Completed processing {frame_count} frames.")
    print(f"[SUCCESS] Result saved to: {os.path.join(project, name)}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Multi-Object Tracking with ByteTrack")
    parser.add_argument("--source", type=str, default="videos/Normal_Videos_606_x264.mp4", help="Path to input video")
    parser.add_argument("--weights", type=str, default="weights/yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config (bytetrack.yaml or botsort.yaml)")
    parser.add_argument("--project", type=str, default="runs/track", help="Output project directory")
    parser.add_argument("--name", type=str, default="hasil_deteksi", help="Run name")
    args = parser.parse_args()

    run_tracking(
        video_path=args.source,
        weights_path=args.weights,
        conf=args.conf,
        tracker=args.tracker,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
