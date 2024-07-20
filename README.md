# Emotion Recognition
This repo contains the code for the paper: 

***MTLSER: Multi-Task Learning Enhanced Speech Emotion Recognition with Pre-trained Acoustic Model***

The code is based on https://github.com/huggingface/transformers/tree/master/examples/research_projects/wav2vec2.


## Set up environment
```bash
pip install -r requirements.txt
```


## Prepare datasets
1. Obtain IEMOCAP dataset from https://sail.usc.edu/iemocap/.
2. Extract and save wav files at some path, assuming named as /wav_path/.
3. Replace the '/path_to_wavs' text in ./iemocap/\*.csv, with the actual path just saved all the wav files. You can use the following command.
```bash
for f in iemocap/*.csv; do sed -i 's/\/path_to_wavs/\/wav_path/' $f; done
```


## How to run
```bash
bash run.sh
```

## For inference 
After training, you can run the inference code, using the saved model in output/tmp (or providing another path with a saved model):
```bash
bash prediction.sh output/tmp
```



