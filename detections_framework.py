from luminoth import Detector, read_image

class DetectionsFramework:
	def __init__(self, image_path, checkpoint):
		self.image_path = image_path
		self.checkpoint = checkpoint


	def run(self):
		detector = Detector(self.checkpoint)
		image = read_image(self.image_path)
		return detector.predict(image)