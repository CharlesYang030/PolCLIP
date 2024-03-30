# PolCLIP
PolCLIP: A Unified Image-Text Word Sense Disambiguation Model via Generating Multimodal Complementary Representations (submitted to ACL 2024)

### Abstract:
Word sense disambiguation (WSD) is divided into two subtasks: textual word sense disambiguation (Textual-WSD) and visual word sense disambiguation (Visual-WSD). They aim to identify the most semantically relevant senses or images to a given context containing ambiguous target words. However, existing WSD models seldom address these two subtasks jointly due to lack of images in Textual-WSD datasets or lack of senses in Visual-WSD datasets. To bridge this gap, we propose PolCLIP, a unified image-text WSD model. By employing an image-text complementarity strategy, it simulates stable diffusion to generate implicit visual representations for senses and imitates image captioning to provide implicit textual representations for images. Additionally, a disambiguation-oriented image-sense dataset is constructed for the training objective of learning multimodal polysemy representations. To the best of our knowledge, PolCLIP is the first model that can cope with both Textual-WSD and Visual-WSD. Extensive experimental results on benchmarks demonstrate the effectiveness of our method, achieving a 2.53% F1-score increase over the state-of-the-art models on Textual-WSD and a 2.22% HR@1 improvement on Visual-WSD.


#### The PolCLIP model:
![image](./model.png)

#### Illustration of Multimodal-WSD:
![image](./mwsd.png)
---

### Environment
Our code has been implemented on Pytorch 2.0.1. To reproduce our experiments, please run: <pre/>pip install -r requirements.txt</pre> 

### 1. USAGE
#### Download the datasets: 
(1) If you are interested in our disambiguation-oriented image-sense dataset, you can click the following links to download the different datasets separately.

Datasets | Instance | Instance type | Image | Image Size | Image Link | Metadata Size | Metadata Link
--- | :---: | :---: | :---: | :---: | :---: | :---: | :---:
Image-Enhanced SemCor | 226,036 | Sentence | 181,123 | 195GB | [Download](https://pan.baidu.com/s/1kEdR58u6by0DYSlDdjQylg?pwd=fkbd) | 3.06GB | [Download](https://drive.google.com/file/d/1l5rBOQDXoTbeW0AXQWA8aGTld3O6lQGt/view?usp=sharing)
Image-Enhanced VWSD-KB | 48,469 | Phrase | 111,575 | 108GB | [Download](https://pan.baidu.com/s/1kEdR58u6by0DYSlDdjQylg?pwd=fkbd) | 0.97GB | [Download](https://drive.google.com/file/d/1aRWlUg36IaaBF774CpCyqAtnOh11E-2u/view?usp=sharing)

(2) The benchmarks of Textual-WSD and Visual-WSD.

Download the XL-WSD data at [https://sapienzanlp.github.io/xl-wsd/](https://sapienzanlp.github.io/xl-wsd/).

Download the V-WSD data at [https://raganato.github.io/vwsd/](https://raganato.github.io/vwsd/).

### 2. REPRODUCE OUR WORK
If you don't have GPUs above 24GB, please use `generate_train_data.py` and adjust smaller `gloss_batch_size` (e.g. [10, 20]) and smaller `image_batch_size` (e.g. [50, 100]) to generate smaller `metadata` for training .

(1) To train from the scratch, please run:
```.
python main.py
```

(2) To evaluate using the best checkpoint, please run:
```.
python main.py --use_checkpoint --evaluate 
```

---

### 3. FINE-TUNE D-GPT
For users with OpenAI account:

(1) If you want to reproduce the process of fine-tuning a disambiguation-oriented GPT-3.5 (approximately 175 mins and $42.5), please download the fine-tuning corpus that we used. [Link: https://drive.google.com/file/d/1qdqt9n3pfnJf9nM3eBnhuxXggDv-1ExR/view?usp=sharing](https://drive.google.com/file/d/1qdqt9n3pfnJf9nM3eBnhuxXggDv-1ExR/view?usp=sharing)

(2) You need to upload the fine-tuning corpus to the OpenAI fine-tuning platform ([https://platform.openai.com/finetune](https://platform.openai.com/finetune)). 

You can find simple fine-tuning instructions at the OpenAI official guideline ([https://platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)) and Appendix B in our paper.

(3) After fine-tuning, you will get a D-GPT that is stored in your own OpenAI account.

(4) Before testing, you need to use D-GPT to generate lexical difinition for ambiguous target words within different contexts by following `D-GPT.py`.

For users without OpenAI account:

(1) Our D-GPT model name: `ft:gpt-3.5-turbo-1106:2023python::8RyWR8GA`

(2) Our API key: `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`

(3) Before executing `D-GPT.py`, you can get our fine-tuned D-GPT model by replacing the model name and the API key with our model name and our API key.

---

### Acknowledgement
PolCLIP benifits from [BabelNet](https://babelnet.org/) and [BabelPic](https://sapienzanlp.github.io/babelpic/). The original authors and their open-sourcing are appreciated.
