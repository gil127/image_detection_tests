import pytest
import numpy as np
import sys
import os
from detections_framework import DetectionsFramework

IMAGE_LIB = "./images/"
BICYCLE_IMAGE = "bicycling-1160860_1280.jpg"
FILE_WITH_EXPECTED_RESULT = 'expected_result_for_test_bicycle_image.json'
detector = DetectionsFramework("./images/bicycling-1160860_1280.jpg", 'fast')
objects = detector.run("{image_name}.png".format(image_name=os.path.splitext(BICYCLE_IMAGE)[0]))


def test_detect_multiple_object_with_fast_checkpoint():
	unique_name = sys._getframe().f_code.co_name + '_' + detector.get_current_timestamp_as_string()
	expected = detector.get_expected_from_file(FILE_WITH_EXPECTED_RESULT, "TestBicycleImage.test_detect_multiple_object_with_fast_checkpoint")
	detector.save_test_output(objects, unique_name + ".json")
	assert len(objects) == expected

def test_detect_image_lables_with_fast_checkpoint():
	unique_name = sys._getframe().f_code.co_name + '_' + detector.get_current_timestamp_as_string()
	expected = detector.get_expected_from_file(FILE_WITH_EXPECTED_RESULT, "TestBicycleImage.test_detect_image_lables_with_fast_checkpoint")
	detector.save_test_output(objects, unique_name + ".json")
	for idx, obj in enumerate(objects):
		assert obj['label'] == expected[idx]
	
def test_detect_image_probabilities_with_fast_checkpoint():	
	unique_name = sys._getframe().f_code.co_name + '_' + detector.get_current_timestamp_as_string()
	expected = detector.get_expected_from_file(FILE_WITH_EXPECTED_RESULT, "TestBicycleImage.test_detect_image_probabilities_with_fast_checkpoint")
	detector.save_test_output(objects, unique_name + ".json")
	for idx, obj in enumerate(objects):
		assert obj['prob'] == expected[idx]
	
def test_detect_image_bbox_with_fast_checkpoint():
	unique_name = sys._getframe().f_code.co_name + '_' + detector.get_current_timestamp_as_string()
	expected = detector.get_expected_from_file(FILE_WITH_EXPECTED_RESULT, "TestBicycleImage.test_detect_image_bbox_with_fast_checkpoint")
	detector.save_test_output(objects, unique_name + ".json")
	for idx, obj in enumerate(objects):
		np.array_equal(expected[idx], obj['bbox'])
