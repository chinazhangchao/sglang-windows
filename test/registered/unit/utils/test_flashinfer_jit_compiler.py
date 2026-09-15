import os
import unittest
from unittest.mock import patch

from sglang.srt.environ import configure_flashinfer_jit_compiler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlashInferJitCompilerConfig(CustomTestCase):
    ENV_NAME = "FLASHINFER_EXTRA_CUDAFLAGS"

    def setUp(self):
        previous = os.environ.pop(self.ENV_NAME, None)

        def restore():
            if previous is None:
                os.environ.pop(self.ENV_NAME, None)
            else:
                os.environ[self.ENV_NAME] = previous

        self.addCleanup(restore)

    def test_adds_msvc_permissive_mode_on_windows(self):
        with patch("sglang.srt.environ.sys.platform", "win32"):
            configure_flashinfer_jit_compiler()

        self.assertEqual(
            os.environ[self.ENV_NAME],
            "-Xcompiler=/permissive",
        )

    def test_preserves_existing_flags_and_is_idempotent(self):
        os.environ[self.ENV_NAME] = "--use_fast_math"

        with patch("sglang.srt.environ.sys.platform", "win32"):
            configure_flashinfer_jit_compiler()
            configure_flashinfer_jit_compiler()

        self.assertEqual(
            os.environ[self.ENV_NAME],
            "--use_fast_math -Xcompiler=/permissive",
        )

    def test_does_not_change_flags_on_other_platforms(self):
        os.environ[self.ENV_NAME] = "--use_fast_math"

        with patch("sglang.srt.environ.sys.platform", "linux"):
            configure_flashinfer_jit_compiler()

        self.assertEqual(os.environ[self.ENV_NAME], "--use_fast_math")


if __name__ == "__main__":
    unittest.main()
