# DNABERT2-CAMP

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
- `data/Ecoli_Promoter_256_independent_test.csv`: Independent test dataset, used for model validation
- `data/Ecoli_Promoter_8720_balanced.csv`:The dataset containing 8720 entries, used as base data for model training
- `model/.gitattributes`: Git attributes configuration file, used to define Git's handling rules for files
- `model/bert_layers.py`: Code for defining the layer structure of the BERT model, including the implementation of core network layers
- `model/bert_padding.py`: Code for handling padding logic of BERT model inputs
- `model/config.json`: Core configuration file for the model, storing key information such as model structure and hyperparameters
- `model/configuration_bert.py`: Code for defining the configuration class of the BERT model, used to load and manage model configurations
- `model/flash_attn_triton.py`: Code for Flash Attention (efficient attention mechanism) implemented based on Triton
- `model/generation_config.json`: Configuration file related to model generation tasks, defining generation parameters
- `model/LICENSE.txt`: Project license file, explaining open-source license terms
- `model/pytorch_model.bin`: Model weight file in PyTorch format, storing trained model parameters
- `model/README.md`: Project documentation, containing information such as project introduction and usage methods
- `model/tokenizer.json`: Core configuration file for the Tokenizer, containing information such as the vocabulary
- `model/tokenizer_config.json`: Configuration parameter file for the Tokenizer, defining tokenization rules, etc.
- `dnabert2_camp_train_and_eval.py`: Training and evaluation script for the DNABERT2-CAMP model 
- `dnabert2_camp_attention_viz.py`: Script for visualizing attention weights from the DNABERT2-CAMP and extracting/analyzing significant sequence motifs associated with high CNN activations.


## Usage

### Training and Testing the Model

```bash
python dnabert2_camp_train_and_eval.py
```

### Analyzing Attention Patterns

```bash
python dnabert2_camp_attention_viz.py
```

### Environment Requirements
```bash
Python 3.9+
CUDA support (optional, for accelerating deep learning models)
```