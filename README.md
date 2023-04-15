# Parkinson’s Severity Diagnosis Explainable Model Based on 3D Multi-Head Attention Residual Network

Jiehui Huang, Lishan Lin, Xuedong He, Fengcheng Yu, Wenhui Song, Jiaying Lin, Zhenchao Tang, Kang Yuan, Yucheng Li, Haofan Huang, Wenbiao Xian, Zhong Pei, Calvin Yu-Chian Chen

<hr />

> **Abstract**--The severity evaluation of Parkinson's disease (PD) is of great significance for the treatment of PD, and many methods have been proposed to solve this problem. However, existing methods either have limitations based on prior knowledge or are invasive methods that require wearable devices. To propose a more generalized severity evaluation model, this paper proposes an explainable 3D multi-head attention residual convolution network. First, we use a 3D attention-based convolution layer as the basic network to initially extract video features. Second, these features will be fed into our LSTM and residual backbone network, which can be used to capture the contextual information of the video. Finally, we design a feature compression module to condense the learned contextual features to perform classification tasks. We also develop some interpretable experiments to better explain this black-box model so that it can be generalized. Experiments show that our model can achieve the best 97% accuracy on the test dataset, which can achieve state-of-the-art prediction performance. The proposed lightweight but effective model is expected to serve as a suitable end-to-end deep learning baseline in future research on PD video-based severity evaluation. The source code is available at https://github.com/JackAILab/ MARNet.

***

## Network Architecture

![](figures/img.png) 

## Installation

See [INSTALL.md](INSTALL.md) for the installation of dependencies required to run Restormer.

## Training and Testing

Training and Testing instructions for MARNet. Here is a summary table containing hyperlinks for easy navigation.

| Training Navigation                   | Testing Navigation                  |
| ------------------------------------- | ----------------------------------- |
| [TrainNavigation](TrainNavigation.md) | [TestNavigation](TestNavigation.md) |


## Results

Experiments are performed for different image processing tasks including, image deraining, single-image motion deblurring, defocus deblurring (both on single image and dual pixel data), and image denoising (both on Gaussian and real data). 

<details>
<summary><strong>Classification Prediction Results</strong> (click to expand) </summary>


 <center><img src="figures/img1.png" style="zoom: 33%;" />
</details>


<details>
<summary><strong>Ablation Experiment Results</strong> (click to expand) </summary></details>


<img src="figures/img2.png" style="zoom: 50%;" />


## Contact

Should you have any question, please contact jiehuihuang1107@163.com
