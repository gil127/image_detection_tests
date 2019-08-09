import pytest
import sys
import os
from detections_framework import DetectionsFramework

IMAGE_LIB = "./images/"
CAR_IMAGE = "105118094-GettyImages-926034838.jpg"
CHECKPOINT_TYPE = "fast"
THRESHOLD = 0.00001
FILE_WITH_EXPECTED_RESULT = 'expected_result_for_test_car_image.json'
detector = DetectionsFramework("{}/{}".format(IMAGE_LIB, CAR_IMAGE), CHECKPOINT_TYPE)
objects = detector.run("{image_name}.png".format(image_name=os.path.splitext(CAR_IMAGE)[0]))
	
def test_probs_threshold_with_fast_checkpoint():	
	unique_name = sys._getframe().f_code.co_name + '_' + detector.get_current_timestamp_as_string()
	expected = detector.get_expected_from_file(FILE_WITH_EXPECTED_RESULT, 'TestCarImage.test_probs_threshold_with_fast_checkpoint')
	detector.save_test_output(objects, unique_name + ".json")
	for idx, obj in enumerate(objects):
		assert(expected[idx] - obj['prob'] <= THRESHOLD)
