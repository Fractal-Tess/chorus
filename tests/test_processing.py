import unittest

from chorus.processing import Processor


class ProcessorTests(unittest.TestCase):
    def test_cuda_worker_keeps_its_physical_device(self):
        self.assertEqual(Processor("cuda:1").device, "cuda:1")

    def test_non_cuda_worker_uses_cpu_processing(self):
        self.assertEqual(Processor("cpu").device, "cpu")
        self.assertEqual(Processor("mps").device, "cpu")


if __name__ == "__main__":
    unittest.main()
