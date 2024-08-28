#Model Imports
import transformers
from transformers import BartForConditionalGeneration, BartTokenizerFast
from transformers import get_scheduler
from accelerate import Accelerator
from accelerate.utils import tqdm

#General Imports
import torch
from torch import nn, Tensor, optim
F = nn.functional

#Dataset imports
from torch.utils.data import Dataset, DataLoader
from random import randint
import json
import os

class JsonL(Dataset):
    def __init__(self, datasetPath, tokenizier, alternate=True):
        self.dpath = os.path.expanduser(datasetPath)
        self.alternate = alternate
        self.tokenizer = tokenizier

        self.data = []
        with open(self.dpath, 'r') as file:
            for line in file:
                self.data.append(json.loads(line)) #returns dict
        #TEMPORARY
        self.data = self.data[:500]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        tPair = self.data[index]
        t1, t2 = tPair['tr'], tPair['en']
        if self.alternate:
            if randint(0,1) == 1:
                t1,t2 = t2,t1

        Input = self.tokenizer(t1, return_tensors='pt', padding='max_length', truncation=True)
        Input, Attn = Input['input_ids'].squeeze(), Input['attention_mask'].squeeze()
        Target = self.tokenizer(t2, return_tensors='pt', padding='max_length', truncation=True)['input_ids'].squeeze()
        return Input, Target, Attn

if __name__ == '__main__': #TOKENIZERS_PARALLELISM=true python train.py
    accelerator = Accelerator(log_with="wandb")
    DEVICE = accelerator.device
    print(accelerator.device)

    LR = 2e-5
    EPOCH = 10
    BATCH = 12

    accelerator.init_trackers(
    project_name="En-Tr-Translation", 
    config={
        "learning_rate": LR,
        "architecture": "BART-ForConditionalGeneration",
        "dataset": "En-Tr-Corpus",
        "epochs": EPOCH,
    },
    init_kwargs={"wandb": {"entity": "###"}}
    )

    model = BartForConditionalGeneration.from_pretrained('En-Tr-Model/')
    model.to(DEVICE)
    tokenizer = BartTokenizerFast.from_pretrained('En-Tr-Model/')

    dataset = JsonL('~/Desktop/CompSciNEA/En-Tr-Corpus/data.jsonl', tokenizer)
    dataloader = DataLoader(dataset,batch_size=BATCH, shuffle=True, num_workers=4)

    optimizer = optim.AdamW(model.parameters(), lr=LR) #, weight_decay=0.01)

    # Learning Rate Scheduler
    total_steps = EPOCH * int(len(dataset) / BATCH) #len(dataloader)  # num_epochs * steps per epoch
    scheduler = get_scheduler(
        "linear",#linear
        optimizer=optimizer,
        num_warmup_steps=int(0.10 * total_steps),  # 10% of total steps for warmup
        num_training_steps=total_steps
    )

    dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)


    for epoch in range(EPOCH):
        model.train()
        # print(f'Epoch:{epoch}')

        accelerator.wait_for_everyone()
        model = accelerator.unwrap_model(model)
        model.save_pretrained(f'SavedModel_{epoch}/',is_main_process=accelerator.is_main_process,save_function=accelerator.save,)
        tokenizer.save_pretrained(f'SavedModel_{epoch}/')

        tepoch = tqdm(iterable=dataloader, main_process_only=True)
        for (inT,outT,attn) in tepoch:
            
            loss = model(input_ids=inT, attention_mask=attn, labels=outT).loss
            accelerator.backward(loss)

            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            current_lr = optimizer.param_groups[0]['lr']
            accelerator.log({"loss": loss.item(), "learning_rate": current_lr})

    accelerator.wait_for_everyone()
    accelerator.end_training()

