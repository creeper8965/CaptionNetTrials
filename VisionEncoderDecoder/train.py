import torch
from torch import optim, device, nn
import transformers
from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessorFast #, AlbertTokenizerFast
from transformers import get_scheduler

from torch.utils.data import Dataset, DataLoader
import pandas as pd
from io import BytesIO
import glob
import os
from PIL import Image

from tqdm import tqdm
import wandb


class Flickr8k(Dataset):
    def __init__(self, dataset_path, transforms, tokenizer):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.transforms = transforms
        self.Tokenizer = tokenizer
        self.txt_path = os.path.join(self.dataset_path, 'captions.txt')
        self.images = os.path.join(self.dataset_path, 'images/')
        self.dataframe = pd.read_csv(self.txt_path)
        self.dataframe['image'] = self.dataframe['image'].apply(lambda x: os.path.join(self.images, x))
        assert len(self.dataframe['image']) == len(self.dataframe['caption'])

    def __len__(self):
        return len(self.dataframe['image'])

    def __getitem__(self, idx):
        caption = self.dataframe['caption'][idx]
        processedCaption = self.Tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True)
        tokens = processedCaption['input_ids'].squeeze(0)
        attention = processedCaption['attention_mask'].squeeze(0)
        image = Image.open(self.dataframe['image'][idx]).convert('RGB')
        image = self.transforms(image, return_tensors='pt')['pixel_values'].squeeze(0)
        return (image,tokens,attention)

class Flickr30k(Dataset):
    def __init__(self, dataset_path, transforms, tokenizer):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.transforms = transforms
        self.Tokenizer = tokenizer
        self.txt_path = os.path.join(self.dataset_path, 'captions.txt')
        self.images = os.path.join(self.dataset_path, 'images/')
        self.dataframe = pd.read_csv(self.txt_path)
        self.dataframe['image_name'] = self.dataframe['image_name'].apply(lambda x: os.path.join(self.images, x))
        assert len(self.dataframe['image_name']) == len(self.dataframe['comment'])

    def __len__(self):
        return len(self.dataframe['image_name'])

    def __getitem__(self, idx):
        caption = self.dataframe['comment'][idx]
        processedCaption = self.Tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True)
        tokens = processedCaption['input_ids'].squeeze(0)
        attention = processedCaption['attention_mask'].squeeze(0)
        image = Image.open(self.dataframe['image_name'][idx]).convert('RGB')
        image = self.transforms(image, return_tensors='pt')['pixel_values'].squeeze(0)
        return (image,tokens,attention)

class ParquetDataset(Dataset):
    def __init__(self, dataset_path, transforms, tokenizer):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.transforms = transforms
        self.Tokenizer = tokenizer

        # # Use glob to get all files matching the pattern
        # self.files = sorted(glob.glob(file_pattern, root_dir=self.dataset_path))
        # self.files = [os.path.join(self.dataset_path, i) for i in self.files]
        
        # # Load all parquet files and combine into a single DataFrame
        # df = pd.concat([pd.read_parquet(f) for f in self.files], ignore_index=True)
        df = pd.read_parquet(self.dataset_path)
        
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
        processedCaption = self.Tokenizer(caption, return_tensors='pt', padding='max_length', truncation=True)
        tokens = processedCaption['input_ids'].squeeze(0)
        attention = processedCaption['attention_mask'].squeeze(0)
        image = self.transforms(image.convert('RGB'), return_tensors='pt')['pixel_values'].squeeze(0)
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


def TrainCycle(dataloader):
    for (img,lab,atm) in tqdm(dataloader):
        img, lab, atm = img.to(DEVICE), lab.to(DEVICE), atm.to(DEVICE)

        loss = model(pixel_values=img, labels=lab, decoder_attention_mask=atm).loss
        loss.backward()

        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({"loss": loss.item(), "learning_rate": current_lr})


if __name__ == '__main__': #TOKENIZERS_PARALLELISM=true python train.py
    LR = 2e-5
    EPOCH = 10
    BATCH = 8
    wandb.init(
        # set the wandb project where this run will be logged
        project="VisionEncodeDecodeTransformer_captionnet",

        # track hyperparameters and run metadata
        config={
        "learning_rate": LR,
        "architecture": "ViT-BertForPreTraining",
        "dataset": "FLICKR8K",
        "epochs": EPOCH,
        }
    )

    DEVICE = torch.device('cuda:0')

    tokenizer = AutoTokenizer.from_pretrained('NewModel/')
    imageProcessor = ViTImageProcessorFast.from_pretrained('NewModel/')
    model = VisionEncoderDecoderModel.from_pretrained('NewModel/')

    model.config.decoder_start_token_id = tokenizer.eos_token_id #CHANGE to cls_token_id if BERT elif BART eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(DEVICE)

    # dataset = Flickr8k('~/Desktop/CompSciNEA/flickr8k/',transforms=imageProcessor,tokenizer=tokenizer)
    # dataset = Flickr30k('~/Desktop/CompSciNEA/flickr30k/',transforms=imageProcessor,tokenizer=tokenizer)
    # dataset = ParquetDataset('val-000??-of-00013.parquet', '~/Desktop/CompSciNEA/COCO-2014/', transforms=imageProcessor, tokenizer=tokenizer)
    # dataloader = DataLoader(dataset=dataset, batch_size=BATCH, shuffle=True,num_workers=4)
    parquet_files, datasetlen, datasteplen = ParquetDataset.GetDatasetPrep('~/Desktop/CompSciNEA/COCO-2014/', 'val-000??-of-00013.parquet')

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    # Learning Rate Scheduler
    total_steps = EPOCH * int(datasteplen / BATCH) #len(dataloader)  # num_epochs * steps per epoch
    scheduler = get_scheduler(
        "cosine",#linear
        optimizer=optimizer,
        num_warmup_steps=int(0.10 * total_steps),  # 10% of total steps for warmup
        num_training_steps=total_steps
    )

    for epoch in range(EPOCH):
        model.train()
        print(f'Epoch:{epoch}')

        model.save_pretrained(f'SavedModel_{epoch}/')
        tokenizer.save_pretrained(f'SavedModel_{epoch}/')
        imageProcessor.save_pretrained(f'SavedModel_{epoch}/')

        for pqrFile in parquet_files:
            dataset = ParquetDataset(pqrFile, transforms=imageProcessor, tokenizer=tokenizer)
            dataloader = DataLoader(dataset=dataset, batch_size=BATCH, shuffle=True,num_workers=4)
            TrainCycle(dataloader)

    wandb.finish()
