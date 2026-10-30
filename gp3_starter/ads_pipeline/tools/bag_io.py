"""Read chosen topics from a rosbag2 folder (MCAP or SQLite). Given."""
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def _storage_id(bag_dir):
    info = rosbag2_py.Info().read_metadata(bag_dir, '')
    return info.storage_identifier


def read(bag_dir, topics):
    """{topic: [(stamp_s, msg), ...]} in bag order. stamp_s is the
    message's header stamp when it has one, else the recording time."""
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id=_storage_id(bag_dir)),
                rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    out = {t: [] for t in topics}
    while reader.has_next():
        topic, data, t_rec = reader.read_next()
        msg = deserialize_message(data, get_message(types[topic]))
        h = getattr(msg, 'header', None)
        t = h.stamp.sec + h.stamp.nanosec * 1e-9 if h is not None else t_rec * 1e-9
        out[topic].append((t, msg))
    return out
