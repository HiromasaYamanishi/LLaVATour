from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
import torch
from tqdm import tqdm
import pandas as pd
import os

class LLaVANext:
    def __init__(self,model_name):
        self.model_name = model_name

        if '34' in model_name:
            # bnb_config = BitsAndBytesConfig(
            # load_in_4bit=True,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16
            # )
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_name, 
                                                                       torch_dtype=torch.bfloat16,
                                                                       #quantization_config=bnb_config,
                                                                       device_map='auto',
                                                                       attn_implementation='flash_attention_2',
                                                                       low_cpu_mem_usage=True) 
            self.model.eval()
            
        else:
            self.model = LlavaNextForConditionalGeneration.from_pretrained(model_name, 
                                                                       torch_dtype=torch.bfloat16,
                                                                       low_cpu_mem_usage=True) 
            self.model.cuda()
            
        #self.model.to("cuda")
        #self.model.cuda()
        self.processor = LlavaNextProcessor.from_pretrained(model_name)
        
    @torch.no_grad()
    def inference(self, image_path, prompt):
        #print('image_path', image_path)
        image = Image.open(image_path).convert('RGB')
        if '13' in self.model_name:
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\n{prompt} ASSISTANT:"
        elif '34' in self.model_name:
            prompt = f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n{prompt}<|im_end|><|im_start|>assistant\n"
        
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda")
        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=16)
        return output
    
    def process_images(self, image_paths, prompts):
        results = []
        for image_path, prompt in tqdm(zip(image_paths, prompts)):
            image_path = os.path.join("/home/yamanishi/project/trip_recommend/data/jalan_image_with_caption", image_path)
            result = self.inference(image_path, prompt)
            print(result)
            results.append(result)
        return results
    
def spot_name_prediction(model_name):
    model = LLaVANext(model_name)
    df = pd.read_csv('./data/df_landmark_recog_eval.csv')
    image_paths = df['image_path'].values
    result = model.process_images(image_paths, ['この画像の観光地名を日本語で答えてください' for _ in range(len(image_paths))])
    pd.DataFrame({'image_path': image_paths, 'predicted': result}).to_csv(f'./result/spot_name/{model_name}.csv')
    
if __name__ == '__main__':
    spot_name_prediction('llava-hf/llava-v1.6-34b-hf')