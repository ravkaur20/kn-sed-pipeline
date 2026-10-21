"""The Outputs/ write guard blocks writes under protected roots only (issue #9)."""
import os
import shutil
import tempfile
import unittest

import _bootstrap_paths  # noqa: F401
from _bootstrap_paths import REPO_ROOT

import _outputs_guard as og


class TestOutputsGuard(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.fake_outputs = os.path.realpath(os.path.join(tmp.name, "Outputs"))
        self.other = os.path.realpath(os.path.join(tmp.name, "scratch"))
        os.makedirs(os.path.join(self.fake_outputs, "SN"))
        os.makedirs(self.other)
        self.existing = os.path.join(self.fake_outputs, "SN", "a.txt")
        with open(self.existing, "w") as fh:
            fh.write("x")
        # Protect the fake tree only for this test; it is unprotected again before tmp cleanup.
        og._PROTECTED.append(self.fake_outputs)
        self.addCleanup(og._PROTECTED.remove, self.fake_outputs)

    def test_installed_on_real_outputs(self):
        self.assertTrue(og._INSTALLED)
        self.assertIn(os.path.realpath(os.path.join(REPO_ROOT, "Outputs")), og._PROTECTED)

    def test_reads_allowed(self):
        with open(self.existing) as fh:
            self.assertEqual(fh.read(), "x")
        self.assertTrue(os.path.isdir(os.path.join(self.fake_outputs, "SN")))

    def test_writes_blocked(self):
        new = os.path.join(self.fake_outputs, "SN", "b.txt")
        for path, mode in [(new, "w"), (new, "a"), (new, "x"), (new, "wb"), (self.existing, "r+")]:
            with self.assertRaises(PermissionError):
                open(path, mode)
        with self.assertRaises(PermissionError):
            os.open(new, os.O_WRONLY | os.O_CREAT)
        with self.assertRaises(PermissionError):
            os.makedirs(os.path.join(self.fake_outputs, "SN", "sub"))
        with self.assertRaises(PermissionError):
            os.remove(self.existing)
        with self.assertRaises(PermissionError):
            shutil.copy2(self.existing, new)
        with self.assertRaises(PermissionError):
            shutil.copytree(os.path.join(self.fake_outputs, "SN"), os.path.join(self.fake_outputs, "SN2"))
        with self.assertRaises(PermissionError):
            shutil.rmtree(os.path.join(self.fake_outputs, "SN"))
        with self.assertRaises(PermissionError):
            os.replace(os.path.join(self.other, "nope"), new)
        self.assertEqual(sorted(os.listdir(os.path.join(self.fake_outputs, "SN"))), ["a.txt"])

    def test_makedirs_exist_ok_on_existing_dir(self):
        os.makedirs(os.path.join(self.fake_outputs, "SN"), exist_ok=True)

    def test_copy_out_of_protected_allowed(self):
        dst = os.path.join(self.other, "SN")
        shutil.copytree(os.path.join(self.fake_outputs, "SN"), dst)
        with open(os.path.join(dst, "a.txt"), "a") as fh:
            fh.write("y")
        self.assertTrue(os.path.isfile(self.existing))

    def test_snapshot_diff(self):
        root = self.other
        with open(os.path.join(root, "keep.txt"), "w") as fh:
            fh.write("1")
        with open(os.path.join(root, "gone.txt"), "w") as fh:
            fh.write("1")
        before = og.snapshot(root)
        os.remove(os.path.join(root, "gone.txt"))
        with open(os.path.join(root, "keep.txt"), "a") as fh:
            fh.write("22")
        with open(os.path.join(root, "new.txt"), "w") as fh:
            fh.write("1")
        changes = og.diff_snapshots(before, og.snapshot(root))
        self.assertEqual(
            changes, ["removed  gone.txt", "modified keep.txt", "added    new.txt"]
        )


if __name__ == "__main__":
    unittest.main()
