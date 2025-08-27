import re
import unittest

from diffusion_policy.common.path_util import resolve_path


class DockerDependencyTest(unittest.TestCase):
    """A sanity check for consistent dependencies between the regular LBM and
    the SageMaker/Docker workflow.
    """

    def _get_torch_cuda_versions(self, text_file, pattern):
        with open(text_file, "r") as f:
            lines = f.readlines()
        for line in lines:
            match = re.search(pattern, line)
            if match:
                return {"torch version": match[1], "CUDA version": match[2]}
        return {}

    def test_torch_cuda_version(self):
        """Checks LBM's torch and CUDA versions from `requirements.in` stay the
        same as the pre-built SageMaker PyTorch image. They should be updated
        together if either of them is changed.
        """
        docker_file_path = resolve_path(
            "package://lbm/setup/docker/Dockerfile"
        )
        lbm_requirements_path = resolve_path("package://lbm/requirements.in")

        # The patterns to extract torch and CUDA versions. For example, we
        # would get `2.2.0` and `121` from `torch==2.2.0+cu121` and
        # `pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker`
        # respectively.
        # TODO: This matches the first instance in the requirements.in file,
        # but since multiple ubuntu / python versions are supported, we
        # technically need to determine which version we should be checking
        # against and then do that. Update this test to handle this more
        # complicated situation.
        requirement_pattern = r"torch==(.*?)\+cu(.*?)(?:;|$)"
        docker_pattern = r".*pytorch-training:(.*?)\-.*cu(.*?)\-"

        # Assert they both contain version values and are identical.
        lbm = self._get_torch_cuda_versions(
            lbm_requirements_path, requirement_pattern
        )
        self.assertTrue(lbm)
        docker = self._get_torch_cuda_versions(
            docker_file_path, docker_pattern
        )
        self.assertTrue(docker)
        self.assertEqual(lbm, docker)


if __name__ == "__main__":
    unittest.main()
