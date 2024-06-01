from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-13b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-13b-hf")

prompts = ["<image>\nUSER: この画像に対するレビューを日本語で出力してください" for i in range(10)]
images = [Image.open(f'/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption/函館山_{i}.jpg') for i in range(10)]

inputs = processor(text=prompts, images=images, return_tensors="pt")

# Generate
generate_ids = model.generate(**inputs, max_length=50)
encode_result = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(encode_result)