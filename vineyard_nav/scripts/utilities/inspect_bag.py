from rosbags.highlevel import AnyReader
from pathlib import Path

with AnyReader([Path('/workspaces/dissertation/kg_march_23_ros2')]) as reader:
    dur = (reader.end_time - reader.start_time) / 1e9
    print(f'Duration: {dur:.1f}s')
    print(f'Messages: {reader.message_count}')
    print(f'Topics:')
    for c in reader.connections:
        print(f'  {c.topic:50s} {c.msgcount:>8d} msgs  {c.msgtype}')