import pytest
import numpy as np
import time
from detections_framework import DetectionsFramework

#@pytest.mark.incremental
class TestBicycleImage:
	# open issues: define init, members and setup (so detect will run once) for now can't define more than 1 test and measure time
	# def __init__(self, image_path, checkpoint):
	# 	self.image_path = image_path
	# 	self.checkpoint = checkpoint

	# @pytest.fixture(scope="session", autouse=True)
	# def test_setup(self):
	# 	detector = DetectionsFramework(self.image_path, self.checkpoint)
	# 	objects = detector.run()

	def test_detect_multiple_object_with_fast_checkpoint(self):
	 	detector1 = DetectionsFramework("./images/bicycling-1160860_1280.jpg", 'fast')
	 	objects1 = detector1.run()
	 	assert len(objects1) == 10

	def test_detect_image_lables_with_fast_checkpoint(self):
		start = time.time()
		detector = DetectionsFramework("./images/bicycling-1160860_1280.jpg", 'fast')
		objects = detector.run()
		labels = ['person', 'person', 'bicycle', 'person', 'bicycle', 'bicycle', 'person', 'person', 'bicycle', 'person']
		for idx, obj in enumerate(objects):
			assert obj['label'] == labels[idx]
		end = time.time()
		print(end - start)
		
	# def test_detect_image_probabilities_with_fast_checkpoint(self):
	# 	detector = DetectionsFramework("./images/bicycling-1160860_1280.jpg", 'fast')
	# 	objects = detector.run()
	# 	probs = [0.9996, 0.9907, 0.9776, 0.968, 0.9645, 0.9614, 0.956, 0.8612, 0.8596, 0.7288]
	# 	for idx, obj in enumerate(objects):
	# 		assert obj['prob'] == probs[idx]
	
	# def test_detect_image_bbox_with_fast_checkpoint(self):
	# 	detector = DetectionsFramework("./images/bicycling-1160860_1280.jpg", 'fast')
	# 	objects = detector.run()
	# 	multi_bbox = [[981, 410, 1077, 614], [688, 384, 769, 586], [994, 501, 1074, 649], [761, 418, 869, 746], [421, 446, 579, 794], [744, 570, 872, 769], [732, 412, 861, 603], [443, 356, 586, 755], [472, 567, 551, 792], [721, 413, 830, 760]]
	# 	for idx, obj in enumerate(objects):
	# 		np.array_equal(multi_bbox[idx], obj['bbox'])