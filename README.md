# Multi-level Part-aware Feature Disentangling for Text-based Person Search

### Prerequisites
* Pytorch 1.8
* python 3.8
* GPU Memory>=16G

### Datasets
* CUHK-PEDES: Please visit [Here](http://xiaotong.me/static/projects/person-search-language/dataset.html).
* ICFG-PEDES (New dataset proposed by Zefeng Ding et al.): Please visit [Here](https://github.com/zifyloo/SSAN).

### Preparations
* You can contact me (chinayhchen@gmail.com) for csv files.
* You need to generate tokens to "/data/BERT_encode/" by running "data_process/CUHK_BERT_token_64.py" and "data_process/ICFG_BERT_token_64";

### Training MPFD and TIPCB (backbone)
* ResNet-BERT structure of MPFD on CUHK-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'ResNet50' --embedding_type 'BERT' --feature_size 4096 --num_epoches 80 --epoches_decay '50' --gpus 0
``
* ResNet-BERT structure of MPFD on ICFG-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'ResNet50' --embedding_type 'BERT' --feature_size 4096 --num_epoches 80 --epoches_decay '50' --gpus 0
``
* ResNet-BERT structure of TIPCB on CUHK-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'ResNet50' --embedding_type 'BERT' --feature_size 2048 --num_epoches 80 --epoches_decay '50' --gpus 0
``
* ResNet-BERT structure of TIPCB on ICFG-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'ResNet50' --embedding_type 'BERT' --feature_size 2048 --num_epoches 80 --epoches_decay '50' --gpus 0
``
* ResNet-LSTM structure of MPFD on CUHK-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'ResNet50' --embedding_type 'Bi-LSTM' --feature_size 4096 --num_epoches 50 --epoches_decay '30' --gpus 0
``
* ResNet-LSTM structure of MPFD on ICFG-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'ResNet50' --embedding_type 'Bi-LSTM' --feature_size 4096 --num_epoches 50 --epoches_decay '30' --gpus 0
``
* ResNet-LSTM structure of TIPCB on CUHK-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'ResNet50' --embedding_type 'Bi-LSTM' --feature_size 2048 --num_epoches 60 --epoches_decay '20_40' --gpus 0
``
* ResNet-LSTM structure of TIPCB on ICFG-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'ResNet50' --embedding_type 'Bi-LSTM' --feature_size 2048 --num_epoches 60 --epoches_decay '20_40' --gpus 0
``
* ResNet-GRU structure of MPFD on CUHK-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'ResNet50' --embedding_type 'Bi-GRU' --feature_size 4096 --num_epoches 50 --epoches_decay '30' --gpus 0
``
* ResNet-GRU structure of MPFD on ICFG-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'ResNet50' --embedding_type 'Bi-GRU' --feature_size 4096 --num_epoches 50 --epoches_decay '30' --gpus 0
``
* ResNet-GRU structure of TIPCB on CUHK-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'ResNet50' --embedding_type 'Bi-GRU' --feature_size 2048 --num_epoches 60 --epoches_decay '20_40' --gpus 0
``
* ResNet-GRU structure of TIPCB on ICFG-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'ResNet50' --embedding_type 'Bi-GRU' --feature_size 2048 --num_epoches 60 --epoches_decay '20_40' --gpus 0
``
* VGG-BERT structure of MPFD on CUHK-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'VGG16' --embedding_type 'BERT' --feature_size 1024 --num_epoches 80 --epoches_decay '50' --gpus 0
``
* VGG-BERT structure of MPFD on ICFG-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'VGG16' --embedding_type 'BERT' --feature_size 1024 --num_epoches 80 --epoches_decay '50' --gpus 0
``
* VGG-BERT structure of TIPCB on CUHK-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'VGG16' --embedding_type 'BERT' --feature_size 512 --num_epoches 80 --epoches_decay '50' --gpus 0
``
* VGG-BERT structure of TIPCB on ICFG-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'VGG16' --embedding_type 'BERT' --feature_size 512 --num_epoches 80 --epoches_decay '50' --gpus 0
``
* VGG-LSTM structure of MPFD on CUHK-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'VGG16' --embedding_type 'Bi-LSTM' --feature_size 1024 --num_epoches 50 --epoches_decay '30' --gpus 0
``
* VGG-LSTM structure of MPFD on ICFG-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'VGG16' --embedding_type 'Bi-LSTM' --feature_size 1024 --num_epoches 50 --epoches_decay '30' --gpus 0
``
* VGG-LSTM structure of TIPCB on CUHK-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'VGG16' --embedding_type 'Bi-LSTM' --feature_size 512 --num_epoches 60 --epoches_decay '20_40' --gpus 0
``
* VGG-LSTM structure of TIPCB on ICFG-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'VGG16' --embedding_type 'Bi-LSTM' --feature_size 512 --num_epoches 60 --epoches_decay '20_40' --gpus 0
``
* VGG-GRU structure of MPFD on CUHK-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'VGG16' --embedding_type 'Bi-GRU' --feature_size 1024 --num_epoches 50 --epoches_decay '30' --gpus 0
``
* VGG-GRU structure of MPFD on ICFG-PEDES:
``
python train_model.py --model 'MPFD' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'VGG16' --embedding_type 'Bi-GRU' --feature_size 1024 --num_epoches 50 --epoches_decay '30' --gpus 0
``
* VGG-GRU structure of TIPCB on CUHK-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'CUHKPEDES' --num_classes 11003 --visual_CNN 'VGG16' --embedding_type 'Bi-GRU' --feature_size 512 --num_epoches 60 --epoches_decay '20_40' --gpus 0
``
* VGG-GRU structure of TIPCB on ICFG-PEDES:
``
python train_model.py --model 'TIPCB' --dataset 'ICFGPEDES' --num_classes 3102 --visual_CNN 'VGG16' --embedding_type 'Bi-GRU' --feature_size 512 --num_epoches 60 --epoches_decay '20_40' --gpus 0
``

### Performance of MPFD
* Performance on CUHK-PEDES

| Structures | Rank-1 | Rank-5 | Rank-10 |
| :------: | :------: | :------: | :------: |
| ResNet-BERT | 66.11 | 84.05 | 90.24 |
| ResNet-GRU | 58.27 | 78.79 | 85.93 |
| ResNet-LSTM | 63.43 | 82.94 | 88.97 |
| VGG-BERT | 60.49 | 80.86 | 87.49 |
| VGG-GRU | 55.36 | 76.80 | 84.49 |
| VGG-LSTM | 58.85 | 79.39 | 86.44 |

* Performance on ICFG-PEDES

| Structures | Rank-1 | Rank-5 | Rank-10 |
| :------: | :------: | :------: | :------: |
| ResNet-BERT | 57.29 | 75.84 | 82.35 |
| ResNet-GRU | 48.39 | 68.48 | 76.61 |
| ResNet-LSTM | 56.58 | 75.32 | 81.90 |
| VGG-BERT | 51.49 | 72.40 | 79.69 |
| VGG-GRU | 47.69 | 68.40 | 76.71 |
| VGG-LSTM | 52.51 | 72.73 | 80.08 |
