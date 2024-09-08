import os
import cv2
import numpy as np
import csv
from PIL import Image
import pickle
from typing import List
import torch

from SeqCLR_RenaissanceSpanish.encoder import Encoder
from SeqCLR_RenaissanceSpanish.Decoder import LSTMAttnDecoder

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Lambda, Grayscale

import math
from typing import Tuple, Union
from deskew import determine_skew

def rotate(image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

data_transform = transforms.Compose([
            Lambda(lambda img: img.convert("RGB")),
            Grayscale(num_output_channels=3),
            Resize((64, 384)),
            ToTensor(),
        ])

with open("SeqCLR_RenaissanceSpanish/Tokenizer/token_to_char.pkl", "rb") as f:
    token_to_char = pickle.load(f)

with open("SeqCLR_RenaissanceSpanish/Tokenizer/char_to_token.pkl", 'rb') as f:
    char_to_token = pickle.load(f)
SOS_token = char_to_token['<SOS>']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder().from_pretrained("yamanoko/SeqCLR_Encoder_fine_tuned")
decoder = LSTMAttnDecoder(256, len(token_to_char)).from_pretrained("yamanoko/SeqCLR_Decoder_fine_tuned")
encoder = encoder.to(device)
decoder = decoder.to(device)

def tokenizer_func(input_list: List[torch.tensor]) -> List[str]:
    output = [token_to_char[index.item()] for index in input_list]
    output = "".join(output)
    output = output.split("<PAD>")[0]
    output = output.split("<EOS>")[0]
    return output

if __name__ == '__main__':
	input_images = os.listdir("inputs")
	for image in input_images:
		img = cv2.imread("inputs/" + image)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		_, threshhold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
		angle = determine_skew(threshhold)
		rotated = rotate(gray, angle, (0, 0, 0))
		cv2.imwrite("inputs/" + image, rotated)
	if not os.path.exists("bbox_result"):
		os.mkdir("bbox_result")
	os.system('python CRAFT-pytorch/test.py --trained_model="CRAFT-pytorch/craft_mlt_25k.pth" --test_folder="inputs" --text_threshold=0.9 --result_folder="bbox_result/"')
	bbox_texts = os.listdir("bbox_result")
	for bbox_text in bbox_texts:
		lines = []
		with open("bbox_result/" + bbox_text, "r") as f:
			reader = csv.reader(f, delimiter=",")
			bboxes = list(reader)
			word_heights = np.array([int(bbox[5]) - int(bbox[3]) for bbox in bboxes])
			word_height = int(np.median(word_heights).item())
			centers = [(int(bbox[1]) + int(bbox[3]) + int(bbox[5]) + int(bbox[7])) // 4 for bbox in bboxes]
			bboxes = [x for _, x in sorted(zip(centers, bboxes), key=lambda x: x[0])]
			line = [np.array(bboxes[0], dtype=np.int32).reshape(-1, 2)]
			for i in range(1, len(bboxes)):
				if int(centers[i] - centers[i-1]) < word_height:
					line.append(np.array(bboxes[i], dtype=np.int32).reshape(-1, 2))
				else:
					line_centers = [(int(bbox[0][0].item()) + int(bbox[1][0].item()) + int(bbox[2][0].item()) + int(bbox[3][0].item())) // 4 for bbox in line]
					line = [x for _, x in sorted(zip(line_centers, line), key=lambda x: x[0])]
					lines.append(line)
					line = [np.array(bboxes[i], dtype=np.int32).reshape(-1, 2)]
			line_centers = [(int(bbox[0][0].item()) + int(bbox[1][0].item()) + int(bbox[2][0].item()) + int(bbox[3][0].item())) // 4 for bbox in line]
			line = [x for _, x in sorted(zip(line_centers, line), key=lambda x: x[0])]
			lines.append(line)
			image = cv2.imread("inputs/" + bbox_text.split(".")[0] + ".png")
		sentences = []
		for line in lines:
			tensor_list = []
			for word_bbox in line:
				width = int(max(np.linalg.norm(word_bbox[1] - word_bbox[0]), np.linalg.norm(word_bbox[3] - word_bbox[2])))
				height = int(max(np.linalg.norm(word_bbox[2] - word_bbox[1]), np.linalg.norm(word_bbox[3] - word_bbox[0])))
				dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
				M = cv2.getPerspectiveTransform(word_bbox.astype(np.float32), dst)
				warped = cv2.warpPerspective(image, M, (width, height))
				warped = Image.fromarray(warped)
				warped = data_transform(warped)
				tensor_list.append(warped)
			with torch.no_grad():
				predictions = []
				tensor_list = torch.stack(tensor_list).to(device)
				encoder_output, decoder_hidden = encoder(tensor_list)
				decoder_input = torch.tensor([[SOS_token]] * tensor_list.shape[0]).to(device)
				for t in range(20):
					decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
					topv, topi = decoder_output.topk(1)
					decoder_input = topi.detach()
					decoder_output_labels = decoder_input.transpose(0, 1)
					predictions.append(decoder_output_labels)
				predictions = torch.cat(predictions, dim=0).transpose(0, 1)
				decoded_predictions = [tokenizer_func(prediction) for prediction in predictions]
				sentence = " ".join(decoded_predictions)
			sentences.append(sentence)
		if not os.path.exists("outputs"):
			os.mkdir("outputs")
		with open("outputs/" + bbox_text.split(".")[0] + ".txt", "w") as f:
			for sentence in sentences:
				f.write(sentence)
				f.write("\n")




