from typing import Optional, Union

import pandas as pd
import typer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.dataset import TestCollateFn, ThisDataset
from src.train_model import LightningModel

"""
python -m src.test_model
"""

def test(
    df_path: str = "data/test.csv", 
    model_path: str =  "google/mt5-base", 
    tokenizer_path: str =  "google/mt5-base",

    ckpt_path: Optional[Union[str, None]] = "models/mt5-base-new/epoch=66-val_loss=1.34124.ckpt",
):
    test_df = pd.read_csv(df_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    test_dataset = ThisDataset(test_df, random=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=14, collate_fn = TestCollateFn(tokenizer, prefix="translate Russian to Lezghian: ",  num_beams=1))

    logger = TensorBoardLogger("./tb_logs", version="mt5-base", name = "test")

    lightning_model = LightningModel(model, tokenizer)
    

    trainer = Trainer(logger=logger, devices = [0], log_every_n_steps=1, precision="32-true")
    trainer.test(model=lightning_model, dataloaders=test_dataloader, ckpt_path=ckpt_path)
    # lightning_model.convert_ckpt_to_tranformers("models/mt5-base-new/converted")

if __name__ == "__main__":
    typer.run(test)