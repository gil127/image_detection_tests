from luminoth import Detector, read_image, vis_objects
from datetime import datetime
import json

class DetectionsFramework:
	def __init__(self, image, checkpoint):
		self.image = image
		self.checkpoint = checkpoint

	def run(self, save_path):
		detector = Detector(self.checkpoint)
		image = read_image(self.image)
		objects = detector.predict(image)
		vis_objects(image, objects).save(save_path)
		return objects

	def get_current_timestamp_as_string(self):
		dateTimeObj = datetime.now()
		timeObj = dateTimeObj.time()
		return timeObj.strftime("%H:%M:%S.%f")

	def get_expected_from_file(self, expected_file, expected_field):
		# TODO: refactor not to read all file
		with open('./' + expected_file) as json_file:
			data = json.load(json_file)
		return (data[expected_field])
	
	def save_test_output(self, data, file_name):
		with open('./outputs/' + file_name, 'w') as outfile:
			json.dump(data, outfile)
