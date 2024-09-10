import os, shutil, glob
import cv2
import numpy as np
import csv
from PIL import Image
import pickle
from typing import List
import torch

from Model.encoder import Encoder
from Model.Decoder import LSTMAttnDecoder

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Lambda, Grayscale

import math
from typing import Tuple, Union
from deskew import determine_skew

with open("Model/Tokenizer/token_to_char.pkl", "rb") as f:
    token_to_char = pickle.load(f)

with open("Model/Tokenizer/char_to_token.pkl", 'rb') as f:
    char_to_token = pickle.load(f)

SOS_token = char_to_token['<SOS>']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder().from_pretrained("yamanoko/SeqCLR_Encoder_fine_tuned")
decoder = LSTMAttnDecoder(256, len(token_to_char)).from_pretrained("yamanoko/SeqCLR_Decoder_fine_tuned")
encoder = encoder.to(device)
decoder = decoder.to(device)

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

def tokenizer_func(input_list: torch.tensor, token_to_char: dict) -> str:
    output = [token_to_char[index.item()] for index in input_list]
    output = "".join(output)
    output = output.split("<PAD>")[0]
    output = output.split("<EOS>")[0]
    return output

def threshholding(image: np.ndarray) -> np.ndarray:
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	_, threshhold = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
	angle = determine_skew(threshhold)
	rotated = rotate(gray, angle, (0, 0, 0))
	return rotated

def image_preprocessing(input_files: List[str], output_dir: str) -> None:
	for input_images in input_files:
		img = cv2.imread(input_images)
		processed_image = threshholding(img)
		cv2.imwrite(output_dir + "/" + os.path.basename(input_images), processed_image)

def extract_bbox(input_dir: str, output_dir: str) -> None:
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	os.system(f'python CRAFT-pytorch/test.py --trained_model="CRAFT-pytorch/craft_mlt_25k.pth" --test_folder="{input_dir}" --text_threshold=0.9 --result_folder="{output_dir}/"')

def segment_line(input_file_path: str) -> List[List[np.ndarray]]:
	lines = []
	with open(input_file_path, "r") as f:
		reader = csv.reader(f, delimiter=",")
		bboxes = list(reader)
		bboxes = list(filter(None, bboxes))
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
	return lines

def crop_word_from_image(bbox: np.ndarray, image: np.ndarray) -> np.ndarray:
	width = int(max(np.linalg.norm(bbox[1] - bbox[0]), np.linalg.norm(bbox[3] - bbox[2])))
	height = int(max(np.linalg.norm(bbox[2] - bbox[1]), np.linalg.norm(bbox[3] - bbox[0])))
	dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
	M = cv2.getPerspectiveTransform(bbox.astype(np.float32), dst)
	warped = cv2.warpPerspective(image, M, (width, height))
	return warped

def make_batch(line: List[np.ndarray], image: np.ndarray) -> torch.Tensor:
	data_transform = transforms.Compose([
            Lambda(lambda img: img.convert("RGB")),
            Grayscale(num_output_channels=3),
            Resize((64, 384)),
            ToTensor(),
        ])
	tensor_list = []
	for word_bbox in line:
		warped = crop_word_from_image(word_bbox, image)
		warped = Image.fromarray(warped)
		warped = data_transform(warped)
		tensor_list.append(warped)
	return torch.stack(tensor_list).to(device)

def recognize_line(line_batch: torch.Tensor) -> str:
	with torch.no_grad():
		predictions = []
		encoder_output, decoder_hidden = encoder(line_batch)
		decoder_input = torch.tensor([[SOS_token]] * line_batch.shape[0]).to(device)
		for t in range(20):
			decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_output)
			topv, topi = decoder_output.topk(1)
			decoder_input = topi.detach()
			predictions.append(decoder_input)
	predictions = torch.cat(predictions, dim=1)
	decoded_predictions = [tokenizer_func(prediction, token_to_char) for prediction in predictions]
	sentence = " ".join(decoded_predictions)
	return sentence

def clear_tmp() -> None:
	try:
		shutil.rmtree("tmp")
	except FileNotFoundError:
		print("Directory does not exist")
	os.mkdir("tmp")
	os.mkdir("tmp/processed_image")
	os.mkdir("tmp/bbox_result")

def inference(input_files: List[str], output_dir: str) -> None:
	clear_tmp()
	processed_image_dir = "tmp/processed_image"
	bbox_result_dir = "tmp/bbox_result"
	image_preprocessing(input_files=input_files, output_dir=processed_image_dir)
	extract_bbox(input_dir=processed_image_dir, output_dir=bbox_result_dir)
	bbox_texts = os.listdir(bbox_result_dir)
	for bbox_text in bbox_texts:
		lines = segment_line(bbox_result_dir + "/" + bbox_text)
		image = cv2.imread(processed_image_dir + "/" + bbox_text.split(".")[0] + ".png")
		sentences = []
		for line in lines:
			line_batch = make_batch(line, image)
			sentence = recognize_line(line_batch)
			sentences.append(sentence)
		with open(output_dir + "/" + bbox_text.split(".")[0] + ".txt", "w", encoding="utf-8") as f:
			for sentence in sentences:
				f.write(sentence)
				f.write("\n")

if __name__ == "__main__":
	image_files = glob.glob(r"C:\Users\yaman\Downloads\Ezcaray\Ezcaray\raw\*.png")
	inference(image_files, r"C:\Users\yaman\Downloads\test_output")
	# bbox_texts = os.listdir("tmp/bbox_result")
	# lines = segment_line("tmp/bbox_result/" + bbox_texts[0])