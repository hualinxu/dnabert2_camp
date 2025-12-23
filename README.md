# DNABERT2-CMAP

## Environment Setup

### Prerequisites
- Anaconda or Miniconda installed
- CUDA-capable GPU (recommended for training)

### Installation Steps

1. Clone this repository
   ```bash
   git clone https://github.com/hualinxu/dnabert2_camp.git
   cd dnabert2_camp
   ```

2. Create and activate the conda environment

   For Windows users:
   ```bash
   # Create environment from the provided YAML file
   conda env create -f environment-win.yml
   
   # Activate the environment
   conda activate myenv
   ```

   For Linux users:
   ```bash
   # Create environment from the provided YAML file
   conda env create -f environment-linux.yml
   
   # Activate the environment
   conda activate myenv
   ```

3. (Optional) Verify environment installation
   ```bash
   conda env list  # Should show 'myenv' as an available environment
   ```

## Code Architecture

### Project Directory Structure and File Descriptions
- `data/independent_test.csv`: Independent test dataset, used for model validation
- `data/original_dataset(8720).csv`: Original dataset containing 8720 entries, used as base data for model training
- `src/model/.gitattributes`: Git attributes configuration file, used to define Git's handling rules for files
- `src/model/bert_layers.py`: Code for defining the layer structure of the BERT model, including the implementation of core network layers
- `src/model/bert_padding.py`: Code for handling padding logic of BERT model inputs
- `src/model/config.json`: Core configuration file for the model, storing key information such as model structure and hyperparameters
- `src/model/configuration_bert.py`: Code for defining the configuration class of the BERT model, used to load and manage model configurations
- `src/model/flash_attn_triton.py`: Code for Flash Attention (efficient attention mechanism) implemented based on Triton
- `src/model/generation_config.json`: Configuration file related to model generation tasks, defining generation parameters
- `src/model/LICENSE.txt`: Project license file, explaining open-source license terms
- `src/model/pytorch_model.bin`: Model weight file in PyTorch format, storing trained model parameters
- `src/model/README.md`: Project documentation, containing information such as project introduction and usage methods
- `src/model/tokenizer.json`: Core configuration file for the Tokenizer, containing information such as the vocabulary
- `src/model/tokenizer_config.json`: Configuration parameter file for the Tokenizer, defining tokenization rules, etc.
- `src/model/DNABERT-2_CNN_test.py`: Test script for the DNABERT-2 combined with CNN model
- `src/model/DNABERT-2_CNN_train_and_eval3.py`: Training and evaluation script for the DNABERT-2 combined with CNN model (version 3)
- `src/model/my_attention3.py`: Implementation code for custom attention mechanism (version 3)


## Usage

### Training the Model

```bash
python DNABERT-2_CNN_train_and_eval3.py
```

### Testing the Model

```bash
python DNABERT-2_CNN_test.py
```
### Analyzing Attention Patterns

```bash
python my_attention3.py
```

### Environment Requirements
```bash
Python 3.9+
CUDA support (optional, for accelerating deep learning models)
```