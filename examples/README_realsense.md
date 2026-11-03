# WAPR-realsense application examples

The maintained entry point is now:

```bash
python -m wapr_realsense --config configs/box.json
```

`realsense_pose.py` is a compatibility launcher. Existing ROI commands continue to work;
the defaults are now ROI initialization, TensorRT and 640×480/15 Hz.
Automatic detection is outside this application's supported workflow.

- [English overview](../README.md) / [中文说明](../README.zh-CN.md)
- [Installation and offline replay](../docs/install.md)
- [Robot interface and read_pose.py](../docs/robotics.md)
- [Jetson deployment notes](../docs/jetson.md)

The original numbered research examples retained in the working checkout belong to the upstream
WAPR release. They are not included in the clean application source export; see
[archived upstream documentation](../docs/upstream/README.md).
