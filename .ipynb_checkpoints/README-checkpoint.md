# TITAN_on_Domino
This repository conatins the code and instructiins to run TITAN (Transformer-based pathology Image and Text Alignment Network) on Domino locally. This is based on https://huggingface.co/MahmoodLab/TITAN

### Make sure Hugging Face Token HUGGINFACE_TOKEN is set as a Domino env varaible. 

This is only requiured to download the model to workspace.


### download the model

```bash
python download_model.py
```

the model will be written to /mnt/titan_model

### Set up the model and cache environment:

```bash
export HF_HOME=/mnt/hf_cache
export TRANSFORMERS_CACHE=/mnt/hf_cache
export HF_MODULES_CACHE=/mnt/hf_cache/modules

```
### Copy your custom files to the cache directory:

```bash
cd /mnt/titan_model
mkdir -p /mnt/hf_cache/modules/transformers_modules/titan_model
cp *.py /mnt/hf_cache/modules/transformers_modules/titan_model/
echo "# Auto-generated" > /mnt/hf_cache/modules/transformers_modules/titan_model/__init__
```

### create synthetic test data 

```bash
python create_test_h5.py
```

### run the streamlit app as a Domino App

You can use the synthetic H5 files to test.