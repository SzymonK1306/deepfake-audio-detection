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

> ❗Remember to update local path to dataset and output dir

```bash
docker run --gpus all -v /full/local/path/to/dataset:/workspace/tacotron2/dataset -v /full/local/path/to/output:/workspace/tacotron2/output -it tacotron2-cuda:latest /bin/bash
```

### Update filelists (below example for LJ Speech dataset)

```bash
sed -i 's,DUMMY,dataset/wavs,g' filelists/*.txt
```

### Start training (default hiperparameters)

```bash
python train.py --output_directory=output/outdir --log_directory=output/logdir
```