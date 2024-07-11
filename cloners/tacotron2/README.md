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
> - Change `/absolute/path/to/folder/data`.
> - Load real dataset in `data/dataset`.
> - Change hiperparameters in `data/hparams.txt`.

```bash
docker run --gpus all -v /absolute/path/to/folder/data:/workspace/tacotron2/data -it tacotron2-cuda:latest /bin/bash
```

### Update filelists (below example for LJ Speech dataset)

```bash
sed -i 's,DUMMY,data/dataset/wavs,g' filelists/*.txt
```

### Start training (default hiperparameters)

> ❗Remember to copy full command

```bash
HPARAMS=$(tr -d "\n\r" < data/hparams.txt | tr -s " ") && python train.py --output_directory=data/output/outdir --log_directory=data/output/logdir --hparams=$HPARAMS
```