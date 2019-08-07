import pytest
import numpy as np
import time
from detections_framework import DetectionsFramework

#@pytest.mark.incremental
class TestCarImage:
	# open issues: define init, members and setup (so detect will run once) for now can't define more than 1 test and measure time
	# def __init__(self, image_path, checkpoint):
	# 	self.image_path = image_path
	# 	self.checkpoint = checkpoint

	# @pytest.fixture(scope="session", autouse=True)
	# def test_setup(self):
	# 	detector = DetectionsFramework(self.image_path, self.checkpoint)
	# 	objects = detector.run()

	def test_detect_multiple_object_with_car_checkpoint(self):
	 	detector = DetectionsFramework("./images/105118094-GettyImages-926034838.jpg", 'fast')
	 	objects = detector.run()
	 	print(objects)
	 	#assert len(objects1) == 10