# PolCLIP
PolCLIP: A Unified Image-Text Word Sense Disambiguation Model via Generating Multimodal Complementary Representations (submitted to ACL 2024)

Word sense disambiguation (WSD) is divided into two subtasks: textual word sense disambiguation (Textual-WSD) and visual word sense disambiguation (Visual-WSD). They aim to identify the most semantically relevant senses or images to a given context containing ambiguous target words. However, existing WSD models seldom address these two subtasks jointly due to lack of images in Textual-WSD datasets or lack of senses in Visual-WSD datasets. To bridge this gap, we propose PolCLIP, a unified image-text WSD model. By employing an image-text complementarity strategy, it simulates stable diffusion to generate implicit visual representations for senses and imitates image captioning to provide implicit textual representations for images. Additionally, a disambiguation-oriented image-sense dataset is constructed for the training objective of learning multimodal polysemy representations. To the best of our knowledge, PolCLIP is the first model that can cope with both Textual-WSD and Visual-WSD. Extensive experimental results on benchmarks demonstrate the effectiveness of our method, achieving a 2.53% F1-score increase over the state-of-the-art models on Textual-WSD and a 2.22% HR@1 improvement on Visual-WSD.


#### The PolCLIP model:
![image](./model.png)
#### Illustration of Multimodal-WSD:
![image](./mwsd.png)
---

### Environment
Our code has been implemented on Pytorch 1.8.1. To reproduce our experiments, please run: <pre/>pip install -r requirements.txt</pre> 

### 1. USAGE
#### 1.Download the datasets: 
(1) If you are interested in our disambiguation-oriented image-sense dataset, you can click the following links to download the different datasets separately.

Datasets | Instance | Instance type | Image | Size | Link | Metadata
--- | :---: | :---: | :---: | :---: | :---: | :---:
Image-Enhanced SemCor | 226,036 | Sentence | 181,123 | 195GB | [Download]() | [Download]()
Image-Enhanced VWSD-KB | 48,469 | Phrase | 111,575 | 108GB | [Download]() | [Download]()

(2) The benchmarks of Textual-WSD and Visual-WSD.
Download [the XL-WSD data](https://sapienzanlp.github.io/xl-wsd/).
Download [the V-WSD data](https://raganato.github.io/vwsd/).

#### 2. REPRODUCE OUR WORK
(1) To train from the scratch, please run:
```.
python main.py
```

(2)To evaluate using the best checkpoint, please run:
```.
python main.py --use_checkpoint --evaluate 
```
---

#### 3. FINE-TUNE D-GPT
For users with OpenAI account:
(1) If you want to reproduce the process of fine-tuning a disambiguation-oriented GPT-3.5 (approximately 175 mins and $42.5), please download the fine-tuning corpus that we used [Download]().
(2) You need to upload the fine-tuning corpus to [the OpenAI fine-tuning platform](https://raganato.github.io/vwsd/). You can also find simple fine-tuning instructions [here](https://platform.openai.com/docs/guides/fine-tuning).
(3) After fine-tuning, you will get a D-GPT that can be used to generate lexical definition for ambiguous target words within different contexts.
