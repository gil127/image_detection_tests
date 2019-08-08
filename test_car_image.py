import pytest
import sys
from detections_framework import DetectionsFramework

#@pytest.mark.incremental
class TestCarImage:
	THRESHOLD = 0.00001
	FILE_WITH_EXPECTED_RESULT = 'expected_result_for_test_car_image.json'
	# open issues: define init, members and setup (so detect will run once) for now can't define more than 1 test and measure time
	# def __init__(self, image_path, checkpoint):
	# 	self.image_path = image_path
	# 	self.checkpoint = checkpoint

	# @pytest.fixture(scope="session", autouse=True)
	# def test_setup(self):
	# 	detector = DetectionsFramework(self.image_path, self.checkpoint)
	# 	objects = detector.run()

	def test_probs_threshold_with_fast_checkpoint(self):
		detector = DetectionsFramework("./images/105118094-GettyImages-926034838.jpg", 'fast')
		unique_name = sys._getframe().f_code.co_name + '_' + detector.get_current_timestamp_as_string()
		objects = detector.run('105118094-GettyImages-926034838_' + unique_name + '.png')
		expected = detector.get_expected_from_file(self.FILE_WITH_EXPECTED_RESULT, 'TestCarImage.test_probs_threshold_with_fast_checkpoint')
		detector.save_test_output(objects, unique_name + ".json")
		for obj in objects:
			assert(expected - obj['prob'] <= self.THRESHOLD)