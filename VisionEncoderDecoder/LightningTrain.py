import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, BlipForConditionalGeneration
import lightning as l

import pandas as pd
from io import BytesIO
import glob
import os
from PIL import Image

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer

torch.set_float32_matmul_precision('medium')#A10 GPU
SavePath = 'MineBLIP/' 

class ParquetDatasetForBLIP(Dataset):
    def __init__(self, dataset_path, transforms):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.transforms = transforms

        # # Use glob to get all files matching the pattern
        self.files = sorted(glob.glob('val-000??-of-00013.parquet', root_dir=self.dataset_path))
        self.files = [os.path.join(self.dataset_path, i) for i in self.files]
        
        # # Load all parquet files and combine into a single DataFrame
        df = pd.concat([pd.read_parquet(f) for f in self.files], ignore_index=True)
        # df = pd.read_parquet(self.dataset_path)
        
        # Process the data into a list of tuples (image, caption)
        self.data = []
        for _, row in df.iterrows():
            # Convert the 'image' field's 'bytes' key into a PIL Image
            image = Image.open(BytesIO(row['image']['bytes']))
            
            # Iterate over each caption and pair it with the image
            for caption in row['answer']:
                self.data.append((image, caption))
        
        df = None
        # longest_string = max(self.data, key=lambda x: len(x[1]))[1]

        # print(f"The longest string is: '{longest_string}'")
        # print(f"It has {len(longest_string)} characters.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, caption = self.data[idx]
        encoding = self.transforms(images=image,text=caption, return_tensors='pt', truncation=True, padding="max_length")
        encoding = {k:v.squeeze() for k,v in encoding.items()} #removes batch
        image = encoding['pixel_values']
        tokens = encoding['input_ids']
        attention = encoding['attention_mask']
        return (image,tokens,attention)

    def GetDatasetPrep(FolderPath, file_pattern):
        path = os.path.expanduser(FolderPath)
        files = sorted(glob.glob(file_pattern, root_dir=path))
        files = [os.path.join(path, i) for i in files]

        total_row = 0
        total_step = 0

        # Calculate the total number of samples across all files
        for file in files:
            df = pd.read_parquet(file)
            total_row += len(df)
            for ans in df['answer']:
                total_step += len(ans)
            df = None

        return files, total_row, total_step


class LitBLIP(l.LightningModule):
    def __init__(self,model):
        super().__init__()
        self.model = model

    def forward(self,inputIds, pixelVals, attention_mask):
        return self.model(input_ids=inputIds,
                        pixel_values=pixelVals,
                        labels=inputIds,
                        attention_mask=attention_mask)

    def training_step(self,batch,batch_idx):
        img, lab, atm = batch
        out = self.forward(lab,img,atm)
        loss = out.loss
        self.log("train_loss", loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        return optimizer

if __name__ == '__main__': #TOKENIZERS_PARALLELISM=true python train.py
    wandb_logger = WandbLogger(log_model="all")

    # Instantiate the Danbooru dataset
    BATCH = 32
    EPOCH = 10

    print('Loading Processor')
    processor = AutoProcessor.from_pretrained(SavePath) #'Salesforce/blip-image-captioning-large'
    print('Loading BLIP')
    blip = BlipForConditionalGeneration.from_pretrained(SavePath)

    print('Making Dataset')
    dataset = ParquetDatasetForBLIP('COCO-2014/', transforms=processor)
    
    print('Model Creation')
    model = LitBLIP(blip)
    model.train()
    trainer = l.Trainer(max_epochs=1,precision='16-mixed',logger=wandb_logger)

    dataloader = DataLoader(dataset=dataset, batch_size=BATCH, shuffle=True,num_workers=4)
    trainer.fit(model, dataloader)


    model.model.save_pretrained(SavePath+'NEW')
    processor.save_pretrained(SavePath+'NEW')
