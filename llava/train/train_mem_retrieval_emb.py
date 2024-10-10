from llava.train.train_retrieve_emb import train
import os
#os.environ["WANDB_PROJECT"]="llava"

if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
