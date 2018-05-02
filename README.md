# SST-Tensorflow

This is a modified version of [JaywongWang's implementation](https://github.com/JaywongWang/SST-Tensorflow). 

Changes I've made:

- The original repo has error in constructing MultiRNN cell, I've fixed it.
- The pretrained model weights given by the original repo have conflicts with the code when loading, so I train the model again and offer my pretrained weights. And I also change the SaverDef version from V1 to V2, as suggested by TensorFlow. By the way, I set the model saving mechanism to only save the best results.
- I add ploting val_loss, val_loss / reg_loss ratio in TensorBoard and create a folder named 'logs' to hold all these tfevents files.
- Learing rate reduce schedule and other minor improvements.

Following is from the readme file in JaywongWang's original repo. Sentences marked with **NOTE** are introduced by me and you should read them carefully.

------------

Tensorflow Implementation of the Paper [SST: Single-Stream Temporal Action Proposals](http://vision.stanford.edu/pdf/buch2017cvpr.pdf) by Shyamal Buch *et al.* in *CVPR* 2017.


### Data Preparation

Please download video data and annotation data from the website [THUMOS14](http://crcv.ucf.edu/THUMOS14/download.html). Extract C3D features for non-overlap 16-frame snippets from the 412 videos (200 val videos + 212 test videos, I found one test video missing) for the task of temporal action proposals. Alternatively, if you don't want to download the dataset and perform the c3d  preprocessing,  you may download my provided [C3D features](https://pan.baidu.com/s/1ggMHZ71), and put them in dataset/thumos14/features/. If you are interested in the feature extraction, I refer you to this [code](https://github.com/yyuanad/Pytorch_C3D_Feature_Extractor).

*fc6* features are used in my experiment.

Please follow the script dataset/thumos14/prepare_gt_proposal_data.py to generate ground-truth proposal data for train/val/test split. I already put the generated files in dataset/thumos14/gt_proposals/.

After that, please generate anchor weights (for handling imbalance class problem) by uniformly sampling video streams (follow dataset/thumos14/anchors/get_anchor_weight.py) or just use my pre-calculated weights (weights.json).


### Hyper Parameters

The best configuration (from my experiments) is given in opt.py, including model setup, training options, and testing options.

**Note**: My config is slightly different from Jay's repo.

### Training

Train your model using the script train.py. Run around 50 epochs and pick the best checkpoint (with the smallest val loss) for prediction.

### Prediction

Follow the script test.py to make proposal predictions.

### Evaluation

Follow the script eval.py to evaluate your proposal predictions.

### Results

You may download my trained model [here](https://pan.baidu.com/s/1mjBI2Nm). Please put them in checkpoints/. Change the file init_from in opt.py and run test.py !

**Note**: Jay's pretrained model fails to load in my experiment. You can use my pretrained model here: [Google Drive link](https://drive.google.com/drive/folders/1dzjeC-B1rDV-gpvBvQRqYN6fwL3sEEIj?usp=sharing). Usage: `python test.py --init_from=checkpoints/1/epoch18_34.54_lr0.001000.ckpt`

**Update:** The predicted action proposals for the test set can be found [here](https://pan.baidu.com/s/1nwa2VLv). The result figures are put in results/1. They are slightly better than the reported ones.

<table>
  <tr>
    <th>Method</th>
    <th>Recall@1000 at tIoU=0.8</th>
  </tr>
  <tr>
    <th>SST (paper)</th>
    <th>0.672</th>
  </tr>
  <tr>
    <th>SST (Jay's impl)</th>
    <th>0.696</th>
  </tr>

 <tr>
    <th>SST (My impl)</th>
    <th>0.697</th>
  </tr>

</table>

![alt text](results/1/sst_recall_vs_proposal.png "Average Recall vs Average Proposal Number")

![alt text](results/1/sst_recall_vs_tiou.png "Recall@1000 vs tIoU")

### Dependencies

tensorflow-gpu(I have tested on tf1.4 and 1.6)

python2

Other versions may also work.

### Acknowledgements

Great thanks to JaywongWang's tensorflow reimplementation.
