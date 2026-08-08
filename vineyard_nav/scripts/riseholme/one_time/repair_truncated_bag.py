"""Rebuild a valid, indexed ROS1 bag from a truncated download. READ-ONLY on the source.

Tue-02-Sep.bag arrived 1.10 GB short of the 44.41 GB its header declares, so its trailing index is
missing and rosbags-convert refuses to open it. Everything present is a clean prefix -- every record
parses to 99.997% of the file's own length -- so the message data can be recovered by walking the
chunks sequentially and written out as a fresh bag with a correct index. The repaired bag then goes
through the SAME standard conversion as every other bag, so the pipeline path is unchanged.

Recovers whole messages only: a partially written trailing chunk is discarded rather than guessed at.

  python3 scripts/riseholme/one_time/repair_truncated_bag.py \
      --src "September 2025/Tue-02-Sep.bag" --dst kg_repaired.bag
"""
import argparse
import struct
import sys
from pathlib import Path

from rosbags.rosbag1 import Writer


def parse_header(b):
    out, i = {}, 0
    while i < len(b):
        n = struct.unpack_from("<I", b, i)[0]; i += 4
        kv = b[i:i + n]; i += n
        k, _, v = kv.partition(b"=")
        out[k.decode()] = v
    return out


def inner_records(buf):
    """Yield (fields, data) for records inside an uncompressed chunk."""
    i, n = 0, len(buf)
    while i + 8 <= n:
        hl = struct.unpack_from("<I", buf, i)[0]; i += 4
        if i + hl + 4 > n:
            return
        fields = parse_header(buf[i:i + hl]); i += hl
        dl = struct.unpack_from("<I", buf, i)[0]; i += 4
        if i + dl > n:
            return
        yield fields, buf[i:i + dl]
        i += dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    if dst.exists():
        raise SystemExit(f"destination already exists: {dst}")
    sz = src.stat().st_size

    f = open(src, "rb")
    if not f.readline().startswith(b"#ROSBAG"):
        raise SystemExit("not a ROS1 bag")

    conns, wconns = {}, {}
    n_chunks = n_msgs = 0
    dropped_partial = 0
    with Writer(dst) as w:
        while True:
            hb = f.read(4)
            if len(hb) < 4:
                break
            hl = struct.unpack("<I", hb)[0]
            hdr = f.read(hl)
            if len(hdr) < hl:
                dropped_partial += 1; break
            db = f.read(4)
            if len(db) < 4:
                dropped_partial += 1; break
            dl = struct.unpack("<I", db)[0]
            fields = parse_header(hdr)
            if fields.get("op", b"\xff")[0] != 5:          # only chunks carry messages
                f.seek(dl, 1); continue
            if f.tell() + dl > sz:                          # partial trailing chunk: discard
                dropped_partial += 1; break
            buf = f.read(dl)
            n_chunks += 1
            for rf, rd in inner_records(buf):
                op = rf.get("op", b"\xff")[0]
                if op == 7:
                    cid = struct.unpack("<I", rf["conn"])[0]
                    if cid in conns:
                        continue
                    topic = rf.get("topic", b"?").decode()
                    ch = parse_header(rd)
                    mt = ch.get("type", b"?").decode()
                    # ROS1 connection headers name types 'pkg/Type'; the writer expects the
                    # normalised 'pkg/msg/Type'. The original string is preserved in msgdef.
                    if mt.count("/") == 1:
                        _pkg, _nm = mt.split("/")
                        mt = f"{_pkg}/msg/{_nm}"
                    conns[cid] = (topic, mt)
                    wconns[cid] = w.add_connection(
                        topic, mt,
                        msgdef=ch.get("message_definition", b"").decode(errors="replace"),
                        md5sum=ch.get("md5sum", b"").decode(errors="replace"),
                        callerid=ch.get("callerid", b"").decode(errors="replace") or None,
                        latching=int(ch["latching"]) if ch.get("latching", b"").isdigit() else None)
                elif op == 2:
                    cid = struct.unpack("<I", rf["conn"])[0]
                    if cid not in wconns:
                        continue
                    sec, nsec = struct.unpack("<II", rf["time"])
                    w.write(wconns[cid], sec * 1_000_000_000 + nsec, rd)
                    n_msgs += 1
            if n_chunks % 2000 == 0:
                print(f"  {n_chunks} chunks, {n_msgs} messages, {100*f.tell()/sz:.1f}% of source",
                      flush=True)
    f.close()
    out_sz = dst.stat().st_size
    print(f"\nrepaired -> {dst}")
    print(f"  chunks read {n_chunks}   messages written {n_msgs}   connections {len(conns)}")
    print(f"  partial trailing records discarded: {dropped_partial}")
    print(f"  source {sz/1e9:.2f} GB -> repaired {out_sz/1e9:.2f} GB")
    print(f"  next:  rosbags-convert --src {dst} --dst <name>_ros2")


if __name__ == "__main__":
    sys.exit(main())
