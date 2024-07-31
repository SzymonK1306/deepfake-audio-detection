# Tacotron2: Docker Image

## General information 

- Make sure you have installed CUDA and Docker

---

## Instruction

### Build Docker image using the provided Dockerfile

```bash
docker build -t tacotron2-cuda:latest .
```

### Run interactive mode with console 

> ❗Remember to :
> - Change `/absolute/path/to/folder/repo`.
> - Load real dataset in `data/dataset`.
> - Change hiperparameters in `data/hparams.txt`.
> - [Optional] Load pre-trained model in `data/models`.

```bash
docker run --gpus all -v /absolute/path/to/folder/repo:/workspace/tacotron2 -it tacotron2-cuda:latest /bin/bash
```

### Update filelists

- LJ Speech dataset

```bash
sed -i 's,DUMMY,data/dataset/wavs,g' filelists/*.txt
```

- OpenSLR (from `repo` folder) - here example for training dataset

```bash
bash ../tools/create_filelists_openslr.sh data/dataset/filelists/audio_text_train_filelist.txt data/dataset/train-clean-100/
```

### Start training (default hiperparameters)

> ❗Remember to copy full command

- Train from zero

```bash
HPARAMS=$(tr -d "\n\r" < data/hparams.txt | tr -s " ") && python train.py --output_directory=data/output/outdir --log_directory=data/output/logdir --hparams=$HPARAMS
```

- Train using a pre-trained model

```bash
HPARAMS=$(tr -d "\n\r" < data/hparams.txt | tr -s " ") && python train.py --output_directory=data/output/outdir --log_directory=data/output/logdir --hparams=$HPARAMS --checkpoint_path=data/models/tacotron2_statedict.pt --warm_start
```

### Check results (examples)

- Own hiperparams

```bash
HPARAMS=$(tr -d "\n\r" < data/hparams.txt | tr -s " ") && python inference.py --input_text "This is so fascinating!" --output_directory=data/output/inference --tacotron_path=data/models/tacotron2_statedict.pt --hparams=$HPARAMS --waveglow_path=data/models/waveglow_256channels_universal_v5.pt
```

- Default hiperparameters

```bash
python inference.py --input_text "This is so fascinating!" --output_directory=data/output/inference --tacotron_path=data/models/tacotron2_statedict.pt --waveglow_path=data/models/waveglow_256channels_universal_v5.pt
```