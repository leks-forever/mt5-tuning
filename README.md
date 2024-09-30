### Install requirements:
```bash
pip install poetry
poetry install
```

### mT5-Tuning


### Raw experimental solution
<!--
Download [bible.csv](https://huggingface.co/datasets/leks-forever/bible-lezghian-russian) and place it in the [data](data) folder. 
-->

Prefixes:    
`"translate Russian to Lezghian: "` - Ru-Lez    
`"translate Lezghian to Russian: "` - Lez-Ru    

Scripts:    
[src/utils.py](src/utils.py) - split prepaired df to train/test/val     
<!--[src/train_tokenizer.py](src/train_tokenizer.py) - update tokenizer and model embeddings according to tokenizer-->     
[src/train_model.py](src/train_model.py) - finetune mT5 model      
[src/test_model.py](src/test_model.py) - test mT5 model  using BLEU and chrF       
<!--[src/predict_model.py](src/predict_model.py) - predict NLLB model-->     
[src/dataset.py](src/dataset.py) - PyTorch train/test datasets

<!--Notebooks:  
[notebooks/convert_to_transformers.ipynb](notebooks/convert_to_transformers.ipynb) -  convert Lighting ckpt to transformers        
[notebooks/predict_model.ipynb](notebooks/predict_model.ipynb) - predict NLLB model
--> 
Logging:
```bash
tensorboard --logdir tb_logs/
```
