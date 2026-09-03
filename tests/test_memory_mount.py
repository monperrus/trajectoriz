"""Tests for the mount-recovery helpers of trajectoriz-cli memory.

A FUSE daemon that dies without unmounting leaves a mountpoint that hangs
every stat() — so detecting and recovering from that state must never itself
touch the filesystem.
"""
from __future__ import annotations

from trajectoriz.cli import _fuse_mountpoints

_MOUNTINFO = """\
23 28 0:22 / /proc rw,nosuid,nodev,noexec,relatime shared:12 - proc proc rw
26 28 0:6 / /dev rw,nosuid,relatime shared:2 - devtmpfs udev rw,size=8129032k
28 1 253:0 / / rw,relatime shared:1 - ext4 /dev/mapper/root rw,errors=remount-ro
99 28 0:55 / /home/u/repo/memory ro,nosuid,nodev,relatime shared:70 - fuse MemoryFS ro
101 28 0:56 / /mnt/with\\040space rw,nosuid,nodev,relatime shared:71 - fuse.sshfs sshfs rw
"""


def test_finds_fuse_mountpoints_only(tmp_path):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(_MOUNTINFO)

    points = _fuse_mountpoints(str(mountinfo))

    assert "/home/u/repo/memory" in points
    assert "/mnt/with space" in points, "octal escapes should be decoded"
    assert "/" not in points and "/proc" not in points


def test_missing_mountinfo_is_not_an_error(tmp_path):
    assert _fuse_mountpoints(str(tmp_path / "nope")) == set()


def test_malformed_lines_are_skipped(tmp_path):
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("garbage\n\n23 28 0:22 /\n" + _MOUNTINFO)

    assert _fuse_mountpoints(str(mountinfo)) == {"/home/u/repo/memory", "/mnt/with space"}
