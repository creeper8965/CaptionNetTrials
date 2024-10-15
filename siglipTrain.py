import transformers
from transformers import SiglipProcessor, SiglipModel, SiglipConfig, SiglipTextConfig, SiglipVisionConfig, SiglipTokenizer
import tqdm
import torch

from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
from torchvision.transforms import v2

import wandb

torch.set_float32_matmul_precision('medium')

class Flickr8k(Dataset):
    def __init__(self, dataset_path, Processor, Transforms, n_samples_to_cache=None):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.Processor = Processor
        self.Transforms = Transforms
        self.txt_path = os.path.join(self.dataset_path, 'captions.txt')
        self.images = os.path.join(self.dataset_path, 'images/')
        self.dataframe = pd.read_csv(self.txt_path)
        self.dataframe['image'] = self.dataframe['image'].apply(lambda x: os.path.join(self.images, x))
        assert len(self.dataframe['image']) == len(self.dataframe['caption'])
        
        self.cache = {}  # Cache dictionary to store preprocessed samples
        self.n_samples_to_cache = n_samples_to_cache if n_samples_to_cache is not None else 0

        # Initialize cache with first n_samples_to_cache
        if self.n_samples_to_cache > 0:
            for idx in range(min(self.n_samples_to_cache, len(self))):
                self.cache[idx] = self._process_sample(idx)

    def _process_sample(self, idx):
        caption = self.dataframe['caption'][idx]
        image = Image.open(self.dataframe['image'][idx])
        inputs = self.Processor(text=caption, images=image, padding='max_length', max_length=64, return_tensors='pt')
        pixels = self.Transforms(inputs['pixel_values'].squeeze(0))
        return pixels, inputs['input_ids'].squeeze(0)

    def __len__(self):
        return len(self.dataframe['image'])

    def __getitem__(self, idx):
        # Return cached sample if available
        if idx in self.cache:
            return self.cache[idx]
        
        # Otherwise, process the sample and return it
        return self._process_sample(idx)
    
class InfiniteDataLoader:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.data_iter = iter(data_loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)  # Reset the data loader
            data = next(self.data_iter)
        return data

if __name__ == '__main__':
    BATCH = 1920 #512
    STEPS = 10000
    DEVICE = 'cuda'
    LR = 3e-3

    wandb.init(
        # set the wandb project where this run will be logged
        project="SigLip",

        # track hyperparameters and run metadata
        config={
        "learning_rate": LR,
        "architecture": "SigLip",
        "dataset": "FLICKR8K",
        "epochs": STEPS,
        }
    )

    Processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    VisionConfig = SiglipVisionConfig(hidden_size=192, intermediate_size=512, num_hidden_layers=12, num_attention_heads=8, num_channels=3, image_size=224, patch_size=16)
    TextConfig = SiglipTextConfig(hidden_size=192, intermediate_size=512, num_hidden_layers=12, num_attention_heads=8)

    ModelConfig = SiglipConfig.from_text_vision_configs(TextConfig, VisionConfig)
    ModelConfig._attn_implementation == "flash_attention_2"
    model = SiglipModel(ModelConfig)
    print('Size:',model.get_memory_footprint() / 1000 /1000,'MB')

    Transforms = v2.Compose([v2.GaussianNoise(),v2.RandomHorizontalFlip()])
    ds = Flickr8k('flickr8k/',Processor, Transforms, 5000)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=4)
    dl = InfiniteDataLoader(dl)

    optimzer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scheduler = transformers.get_scheduler(
        "cosine",#linear
        optimizer=optimzer,
        num_warmup_steps=100,  # 10% of total steps for warmup
        num_training_steps=STEPS
    )

    model.to(DEVICE)
    model.train()
    TT = tqdm.tqdm(iterable=dl, total=STEPS)
    totalSteps = 0
    for pixels, input_ids in TT:
        totalSteps += 1
        optimzer.zero_grad()
        pixels = pixels.to(DEVICE)
        input_ids = input_ids.to(DEVICE)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = model(input_ids=input_ids, pixel_values=pixels, return_loss=True).loss
        loss.backward()
        optimzer.step()
        scheduler.step()
        lossItm = loss.item()
        current_lr = optimzer.param_groups[0]['lr']
        wandb.log({"loss": loss.item(), "learning_rate": current_lr})
        if totalSteps % 100 == 0:
            model.save_pretrained(f'SigLip_{totalSteps}/')
            Processor.save_pretrained(f'SigLip_{totalSteps}/')

    wandb.finish()

    model.save_pretrained('SigLip/')
    Processor.save_pretrained('SigLip/')
