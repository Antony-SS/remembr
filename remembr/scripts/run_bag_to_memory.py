# Example usage function
from remembr.memory.memory_builder import MemoryBuilder
from data_streams.ros2_common.camera_streams import make_rgb_image_stream
from data_streams.collection_streams.research_robot import make_tf_static_to_pose_stream
from tqdm import tqdm
import argparse
from PIL import Image

def build_memory(
    bag_path: str,
    image_topic_name: str,
    pose_topic_name: str,
    collection_name: str,
    db_ip: str,
    db_port: int,
    model_path: str,
    model_base: str,
    frames_per_memory: int,
    skip_frames: int,
    use_header_timestamps: bool,
) -> None:

    builder = MemoryBuilder(
        collection_name=collection_name,
        db_ip=db_ip,
        db_port=db_port,
        model_path=model_path,
        model_base=model_base,
        num_video_frames=frames_per_memory,
    )

    image_stream = make_rgb_image_stream(ros2_mcap_path=bag_path, topic_name=image_topic_name, use_header_timestamps=use_header_timestamps)
    pose_stream = make_tf_static_to_pose_stream(ros2_mcap_path=bag_path, tf_topic_name=pose_topic_name, use_header_timestamps=False)

    if len(image_stream) == 0 or len(pose_stream) == 0:
        raise ValueError(
            f"No images or poses found in bag {bag_path} and topic {image_topic_name} or {pose_topic_name}"
        )

    images_buffer = []
    timestamps_buffer = []
    pose_instances_buffer = []

    for index, image_instance in tqdm(
        enumerate(image_stream.iterate(skip_every=skip_frames)), total=len(image_stream)
    ):
        image = image_instance.data
        index = image_instance.index
        timestamp = image_instance.timestamp

        image = Image.fromarray(image)

        pose_instance = pose_stream.get_nearest_instance(timestamp)

        images_buffer.append(image)
        timestamps_buffer.append(timestamp)
        pose_instances_buffer.append(pose_instance)

        # Only build memory when enough frames have been collected
        if len(images_buffer) == frames_per_memory:
            # For position and yaw, we can use the average over this window
            translations = [pi.pose.translation for pi in pose_instances_buffer]
            yaws = [pi.pose.euler_flu_degrees()[-1] for pi in pose_instances_buffer]
            avg_translation = [
                sum(xs) / len(xs) for xs in zip(*translations)
            ]
            avg_yaw = sum(yaws) / len(yaws)
            avg_time = sum(timestamps_buffer) / len(timestamps_buffer)

            # Passing None for caption lets the builder generate one using VILA
            builder.add_memory(
                images=list(images_buffer),
                position=avg_translation,
                theta=avg_yaw,
                time=avg_time,
                caption=None,
            )

            # Clear buffers for next window
            images_buffer = []
            timestamps_buffer = []
            pose_instances_buffer = []

    # Catch any remaining frames at the end
    if images_buffer:
        translations = [pi.pose.translation for pi in pose_instances_buffer]
        yaws = [pi.pose.euler_flu_degrees()[-1] for pi in pose_instances_buffer]
        avg_translation = [
            sum(xs) / len(xs) for xs in zip(*translations)
        ]
        avg_yaw = sum(yaws) / len(yaws)
        avg_time = sum(timestamps_buffer) / len(timestamps_buffer)
        builder.add_memory(
            images=list(images_buffer),
            position=avg_translation,
            theta=avg_yaw,
            time=avg_time,
            caption=None,
        )

def main():
    args = arg_parser()
    print("Building memory with arguments:")
    print(f"Bag path: {args.bag_path}")
    print(f"Image topic name: {args.image_topic_name}")
    print(f"Pose topic name: {args.pose_topic_name}")
    print(f"Collection name: {args.collection_name}")
    print(f"DB IP: {args.db_ip}")
    print(f"DB port: {args.db_port}")
    print(f"Model path: {args.model_path}")
    print(f"Model base: {args.model_base}")
    print(f"Frames per memory: {args.frames_per_memory}")
    print(f"Skip frames: {args.skip_frames}")
    print(f"Use header timestamps: {args.use_header_timestamps}")
    print("Building memory...")
    build_memory(
        bag_path=args.bag_path,
        image_topic_name=args.image_topic_name,
        pose_topic_name=args.pose_topic_name,
        collection_name=args.collection_name,
        db_ip=args.db_ip,
        db_port=args.db_port,
        model_path=args.model_path,
        model_base=args.model_base,
        frames_per_memory=args.frames_per_memory,
        skip_frames=args.skip_frames,
        use_header_timestamps=args.use_header_timestamps,
    )


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_path", type=str, required=True)
    parser.add_argument("--image_topic_name", type=str, required=False, default="/camera/rgb/image_color")
    parser.add_argument("--pose_topic_name", type=str, default="/pose")
    parser.add_argument("--collection_name", type=str, default="test_collection")
    parser.add_argument("--db_ip", type=str, default="127.0.0.1")
    parser.add_argument("--db_port", type=int, default=19530)
    parser.add_argument("--model_path", type=str, default="Efficient-Large-Model/VILA1.5-13b")  # Instead of VILA1.5-3b,
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--frames_per_memory", type=int, default=6)
    parser.add_argument("--skip_frames", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--use_header_timestamps", type=bool, default=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()