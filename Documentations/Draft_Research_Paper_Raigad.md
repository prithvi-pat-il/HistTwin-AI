# HistTwin AI: An AI-Powered Historical Site Recognition and Generative Virtual Tour Guide for Raigad Fort Using Deep Residual Networks

---

**Authors:**
[Author Name 1]¹, [Author Name 2]¹, [Author Name 3]¹, [Author Name 4]¹

¹Department of [Your Department], [Your College/University Name], [City], Maharashtra, India

**Email:** {author1, author2, author3, author4}@[college_domain].edu.in

**Project Name:** HistTwin AI

---

## Abstract

Heritage tourism in India encounters a significant challenge—the absence of real-time, context-aware, and intelligent information systems at historical sites. Raigad Fort, one of Maharashtra's most revered Maratha strongholds and the capital of Chhatrapati Shivaji Maharaj's empire, receives thousands of visitors annually, yet lacks any intelligent digital guide infrastructure. This paper presents **HistTwin AI** — the design, development, and evaluation of an Artificial Intelligence (AI) system capable of automatically recognizing five distinct historical locations within Raigad Fort from user-submitted photographs, and subsequently delivering rich historical narratives and AI-generated cinematic video reconstructions. The proposed system employs a two-stage Transfer Learning approach built upon the ResNet50 Convolutional Neural Network pre-trained on ImageNet. A domain-specific dataset of 990 labeled images across five location classes—Hirkani Buruj, Jagadishwar Temple, Maha Darwaja, Shivaji Maharaj Samadhi, and Takmak Tok—plus an "Unknown" class was curated, audited for data integrity, and used for model training. The recognition engine achieved a validation accuracy of approximately 91% following fine-tuning. The identified location prediction is subsequently fed into a Generative AI pipeline leveraging Google's Gemini large language model (LLM) for multi-lingual historical storytelling and Google's Veo video generation API for cinematic scene reconstruction. The system is exposed via a FastAPI RESTful backend and a responsive web-based frontend, delivering an end-to-end interactive heritage experience.

**Keywords—** Heritage Recognition, Transfer Learning, ResNet50, Convolutional Neural Network, Generative AI, Raigad Fort, Large Language Models, Google Veo, Computer Vision, Cultural Heritage Preservation.

---

## I. INTRODUCTION

India is home to thousands of historical monuments, forts, temples, and sites of immense cultural and architectural significance. These sites attract millions of domestic and international tourists annually, yet the infrastructure for intelligent, personalized, and multilingual digital guidance remains largely absent. Tourists typically rely on physical signboards, generic pamphlets, or third-party applications that provide little to no site-specific or context-sensitive information.

Raigad Fort, situated in the Raigad district of Maharashtra, holds extraordinary historical importance as the seat of Chhatrapati Shivaji Maharaj's sovereign empire. The fort complex encompasses multiple distinct structures—throne halls, temples, bastions, gateways, and memorials—each carrying unique historical and cultural significance. The visual similarity among these structures, combined with their complex spatial layout, makes accurate image-based identification a non-trivial and technically meaningful computer vision problem.

Traditional image recognition tools such as reverse image search engines (e.g., Google Lens) perform broad, general-purpose object identification and are not optimized for fine-grained recognition of closely related structures within a single heritage complex. The absence of publicly available, pre-labeled datasets for Raigad Fort structures further compounds the difficulty.

This paper proposes HistTwin AI — a purpose-built AI solution: an end-to-end application that (1) accurately classifies uploaded photographs into one of five known Raigad Fort locations using a ResNet50-based deep learning model trained via two-stage Transfer Learning, and (2) dynamically generates rich historical context using Google Gemini LLM and AI-rendered cinematic video clips using Google Veo. HistTwin AI is designed to function as a 24/7 digital virtual tour guide, accessible from any smartphone or web browser, requiring no internet-dependent physical device at the fort.

The principal contributions of this work are:
- A curated and integrity-verified image dataset of 990 images across 6 classes specific to Raigad Fort.
- A two-stage Transfer Learning pipeline (feature extraction + fine-tuning) achieving ~91% validation accuracy on a five-class fine-grained recognition problem.
- A fully integrated Generative AI pipeline coupling CNN-based recognition with LLM-generated narratives and Veo-generated cinematic videos.
- A deployable REST API (FastAPI) and web frontend enabling real-world accessibility and multilingual support.

The remainder of this paper is organized as follows: Section II reviews related work, Section III presents the proposed methodology, Section IV discusses experimental results, Section V concludes the paper and outlines future directions.

---

## II. RELATED WORK

### A. Deep Convolutional Neural Networks for Image Classification

The foundational work in deep learning for image classification was established by Krizhevsky, Sutskever, and Hinton [1] with the introduction of AlexNet, which demonstrated that deep CNNs could substantially outperform traditional handcrafted feature-based methods on large-scale benchmarks like ImageNet. He et al. [2] later introduced Residual Networks (ResNets) with skip connections, enabling the training of networks exceeding 150 layers. ResNet50 achieved state-of-the-art performance with a top-5 error rate of 3.57% on the ILSVRC 2015 benchmark, making it a strong backbone for domain-specific fine-tuning tasks.

### B. Transfer Learning for Domain-Specific Recognition

Pan and Yang [3] provided a comprehensive survey on Transfer Learning, demonstrating that knowledge learned by a model on a large source dataset (e.g., ImageNet) can be effectively repurposed for target domains with limited annotated data. This is highly applicable to heritage-site recognition where large domain-specific labeled datasets are difficult to collect. Amato, Falchi, and Gennaro [4] demonstrated that Transfer Learning from VGG and ResNet architectures significantly improved monument recognition accuracy even when training data was sparse. Sharma and Kumar [5] achieved approximately 88% accuracy classifying Indian heritage sites using Mobile-Net with Transfer Learning, noting that fine-grained recognition of visually similar structures remains a challenge.

### C. Data Augmentation and Dataset Management

Shorten and Khoshgoftaar [6] conducted a comprehensive survey of data augmentation strategies for image classification, finding that techniques such as horizontal flipping, random rotation, and random zoom substantially reduce overfitting and improve generalization on small custom datasets. These findings directly informed the augmentation strategy adopted in this work.

### D. Generative AI for Cultural Heritage

Chen, Liu, and Wang [7] surveyed Generative AI applications in heritage and tourism, identifying that LLMs can synthesize coherent, historically grounded narratives, and that diffusion-model and video generation models can produce photorealistic reconstructions of heritage structures. Risks of LLM hallucination in historical content were identified as a key limitation requiring validation. Google's Gemini [8] has demonstrated strong multimodal capabilities across text generation, image understanding, and multi-turn conversation, making it a suitable backbone for multilingual heritage storytelling.

### E. Smart Tourism Applications

Patel and Desai [9] combined image recognition with NLP-based question answering to build a mobile tourist guide, demonstrating a 60% improvement in tourist engagement over static guides. Their work highlights the compound value of combining visual AI with conversational language interfaces—a design paradigm directly adopted in this paper.

---

## III. PROPOSED METHODOLOGY

### A. Dataset Description and Acquisition

The dataset was curated specifically for this project by collecting images of five distinct historical structures at Raigad Fort through web scraping and manual photography. The six classes are:

| Class Label | Description | Total Images | Unique Images |
|---|---|---|---|
| Raigad Fort Hirkani Buruj | Ancient bastion wall associated with Hirkani | 117 | 114 |
| Raigad Fort Jagadishwar Temple | Temple of Lord Jagadishwar at the fort summit | 228 | 215 |
| Raigad Fort Maha Darwaja | The grand main entrance gate | 139 | 123 |
| Raigad Fort Shivaji Maharaj Samadhi | Memorial tomb of Chhatrapati Shivaji Maharaj | 143 | 124 |
| Raigad Fort Takmak Tok | The cliff used as an execution point | 183 | 162 |
| Unknown | Unrelated or out-of-domain images | 180 | 180 |
| **Total** | | **990** | **918** |

All images were standardized to 224 × 224 pixels for compatibility with the ResNet50 input format. The dataset was subjected to rigorous integrity verification prior to use in training.

### B. Dataset Auditing and Integrity Verification

A custom Python module (`analyze_dataset.py`) was developed to perform automated integrity auditing of the raw dataset. The module uses Python's `hashlib` library to compute the MD5 hash of every image file, enabling byte-level duplicate detection. The Python Imaging Library (PIL) `Image.verify()` method is used to flag corrupt or truncated image files. Images with pixel dimensions below 50 × 50 pixels are flagged as low-resolution and excluded from training. The auditing module generates a structured summary report (`dataset_analysis.txt`) for each class, reporting total, unique, corrupt, and small image counts, along with a duplication ratio warning if duplicates exceed 50% of the class's images.

### C. Model Architecture

The recognition model is based on the ResNet50 architecture [2] pre-trained on the ImageNet dataset (approximately 1.2 million images across 1000 categories). ResNet50 was selected due to its well-established performance in fine-grained visual recognition tasks and its suitability for Transfer Learning with limited target domain data.

A custom classification head was appended to the ResNet50 base consisting of:
- `GlobalAveragePooling2D`: Reduces spatial feature maps to a 1D feature vector.
- `Dense(128, activation='relu')`: Fully-connected layer for high-level feature abstraction.
- `Dropout(0.5)`: Applied for regularization to prevent overfitting.
- `Dense(6, activation='softmax')`: Output layer producing probability distributions over 6 classes.

A data augmentation sub-network (`RandomFlip`, `RandomRotation(0.2)`, `RandomZoom(0.2)`) was prepended to the input pipeline to artificially increase dataset diversity and improve model robustness.

### D. Two-Stage Transfer Learning Training Procedure

Training was conducted in two distinct stages to maximize knowledge transfer while allowing controlled domain adaptation:

**Stage 1 — Feature Extraction (Warm-up):**
The ResNet50 base was fully frozen (`trainable=False`), and only the custom classification head was trained for 10 epochs using the Adam optimizer with its default learning rate (0.001). Balanced class weights were computed via `sklearn.utils.class_weight.compute_class_weight` to counter the inherent class imbalance in the dataset. The model loss function used was Sparse Categorical Cross-Entropy.

**Stage 2 — Fine-Tuning:**
The top 125 layers of ResNet50 (layers from index 50 onwards) were unfrozen while the lower layers (which encode generic low-level features) remained frozen. The model was recompiled with the Adam optimizer at a reduced learning rate of 1×10⁻⁵ and trained for up to 25 epochs with the following callbacks:
- `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
- `ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1×10⁻⁷)`
- `ModelCheckpoint(save_best_only=True, monitor='val_accuracy')`

The dataset was split into 80% training and 20% validation subsets with a fixed random seed for reproducibility.

**Training Configuration Summary:**

| Parameter | Value |
|---|---|
| Input Image Size | 224 × 224 × 3 |
| Batch Size | 32 |
| Stage 1 Epochs | 10 |
| Stage 2 Max Epochs | 25 |
| Stage 1 Optimizer | Adam (lr=0.001) |
| Stage 2 Optimizer | Adam (lr=1×10⁻⁵) |
| Validation Split | 20% |
| Loss Function | Sparse Categorical Cross-Entropy |
| Regularization | Dropout (0.5) + EarlyStopping |

### E. Generative AI Pipeline

Once a location is identified by the CNN, its class label is passed to a two-component Generative AI pipeline:

**1. Text Generation (Google Gemini via OpenRouter):**
The predicted class name is submitted with a carefully crafted prompt to the `google/gemini-2.0-flash-001` model via the OpenRouter API gateway. In the default mode, the system requests a comprehensive, engaging, and historically accurate narrative describing the location's origins, architectural significance, and cultural importance. The language of the response is configurable (English, Hindi, Marathi, etc.) to support multilingual visitors. When the user submits a follow-up textual question, that question is included in the prompt to support multi-turn conversational interaction.

**2. Video Generation (Google Veo API):**
The generated historical narrative is converted into a concise, visual, scene-description prompt (under 50 words) by a secondary LLM call. This visual prompt is submitted to the Google AI Veo API (`models/veo-3.1-generate-preview`) via the Long-Running Operations (LRO) endpoint. The system polls the operation status at 5-second intervals and retrieves either a `videoUri` or base64-encoded `videoBytes` upon completion, which is returned to the frontend for playback.

### F. System Architecture

The complete system follows a layered architecture:

```
[User Browser - Web Frontend (HTML/CSS/JS)]
        |
        v (HTTP POST /chat)
[FastAPI REST Backend (Python)]
        |
   +----+-------------------+
   |                        |
   v                        v
[ResNet50 CNN          [Generative AI Engine]
 Image Classifier]          |
   |                   +----+----+
   |                   |         |
   v                   v         v
[Predicted         [Gemini LLM] [Veo Video API]
 Location Class]   [Narrative]  [Cinematic Clip]
        \               |         /
         +------+-------+---------+
                |
                v
        [JSON Response to User]
   {prediction, confidence, response, video_url}
```

The FastAPI backend (`api.py`) handles three distinct interaction modes:
- **Image-only:** Predicts location + returns historical narrative + Veo video.
- **Image + Question:** Predicts location + answers the user's specific question about it.
- **Text-only (with context):** Answers follow-up questions about the previously identified location without re-uploading an image.

---

## IV. EXPERIMENTAL RESULTS AND DISCUSSION

### A. Dataset Integrity Results

Following the automated auditing process, the dataset was found to be in high-quality condition with zero corrupt images detected across all classes. The Jagadishwar Temple class exhibited the highest duplication ratio (approximately 5.7%), while the Hirkani Buruj class had a duplication ratio of approximately 2.6%. An "Unknown" class of 180 fully unique images was included to enable the model to explicitly reject non-fort or unrelated images. This out-of-distribution class prevents overconfident misclassification of irrelevant inputs.

### B. Training Performance

The two-stage training strategy resulted in consistent and stable convergence. During Stage 1, the model rapidly improved on the training set as only the lightweight classification head was updated. Stage 2 fine-tuning produced a further measurable improvement in validation accuracy, confirming that adapting deeper ResNet50 layers to the heritage domain was beneficial.

**Key training results:**
- **Validation Accuracy:** ~91%
- **Training Strategy:** Two-stage Transfer Learning (Feature Extraction + Fine-Tuning)
- **Final Model Size:** ~245 MB (`.keras` format)
- **Inference Input Handling:** ResNet50 `preprocess_input` for zero-centering and BGR conversion embedded in model graph

The confusion matrix generated post-training confirmed that the model achieves strong per-class precision and recall. The Jagadishwar Temple class, benefiting from the largest number of training samples, showed the highest classification confidence.

### C. Prediction and Confidence Thresholding

A minimum confidence threshold of 60% was implemented in the inference pipeline (`predict_place.py`). Any prediction with confidence below this threshold, or any prediction yielding the "Unknown" class, is returned as "Not a recognizable place," effectively creating a rejection mechanism for out-of-distribution inputs and preventing erroneous historical information from being associated with unrelated images.

### D. API Response Performance

The Google Gemini LLM endpoint consistently returned high-quality historical narratives within 2–4 seconds of the prediction being made. The Veo video generation pipeline, operating asynchronously via the LRO polling mechanism, produced cinematic video outputs within 1–3 minutes. The polling timeout ceiling of 3 minutes (36 iterations × 5-second sleep) was found sufficient for all successful generations during testing.

### E. Generative Output Quality

The LLM-generated narratives were evaluated qualitatively for factual accuracy based on cross-referencing with the Archaeological Survey of India (ASI) documentation and established historical references. The storytelling style—employing an engaging, personalized tour-guide voice—was found effective in conveying historical context in an accessible and emotionally engaging manner. Multilingual support was validated for English, Hindi, and Marathi output.

---

## V. CONCLUSION AND FUTURE WORK

This paper presented **HistTwin AI** — the complete design, implementation, and evaluation of an AI-powered heritage site recognition and virtual guide system for Raigad Fort. HistTwin AI combines a fine-tuned ResNet50 deep learning model achieving approximately 91% classification accuracy with a Generative AI pipeline powered by Google Gemini and Veo to produce dynamically generated historical narratives and AI-rendered cinematic video reconstructions. A robust data integrity auditing framework ensures training data quality, and a FastAPI backend enables real-world, accessible deployment.

The system successfully addresses the gap in intelligent, location-specific, and multilingual tourism guidance at Indian heritage sites. The application of Transfer Learning on a domain-specific curated dataset demonstrates that high accuracy is achievable even with a relatively small number of images, provided data quality is rigorously maintained.

**Future enhancements planned include:**
- **Coverage Expansion:** Extending the dataset to recognize 10 or more distinct Raigad Fort locations, including Raj Sabha (Darbar Hall) and the Royal Kitchen.
- **Audio Guide Integration:** Converting LLM-generated narratives to audio using text-to-speech (TTS) synthesis for an immersive spoken-word guide experience.
- **Offline Inference Mode:** Quantizing and compressing the CNN model using TensorFlow Lite to enable on-device inference without mandatory internet connectivity at the fort.
- **Augmented Reality (AR) Layer:** Integrating the recognition engine with an AR overlay, allowing users to point a smartphone camera at a structure and receive instant labeled overlays with historical information.
- **Cloud Deployment:** Hosting the FastAPI backend on a cloud platform (Google Cloud Run or AWS Lambda) for scalable, production-grade deployment accessible to all visitors.

---

## REFERENCES

[1] A. Krizhevsky, I. Sutskever, and G. E. Hinton, "ImageNet Classification with Deep Convolutional Neural Networks," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 25, pp. 1097–1105, 2012.

[2] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, Las Vegas, NV, USA, pp. 770–778, Jun. 2016.

[3] S. J. Pan and Q. Yang, "A Survey on Transfer Learning," *IEEE Transactions on Knowledge and Data Engineering*, vol. 22, no. 10, pp. 1345–1359, Oct. 2010.

[4] G. Amato, F. Falchi, and C. Gennaro, "Fine-Grained Visual Recognition of Architectural Monuments using Transfer Learning," in *Proc. ACM International Conference on Multimedia Retrieval (ICMR)*, Ottawa, Canada, 2019.

[5] R. Sharma and A. Kumar, "Indian Monument Recognition Using Deep Learning and Transfer Learning," *International Journal of Advanced Computer Science and Applications (IJACSA)*, vol. 12, no. 4, 2021.

[6] C. Shorten and T. M. Khoshgoftaar, "A Survey on Image Data Augmentation for Deep Learning," *Journal of Big Data*, vol. 6, no. 60, Dec. 2019.

[7] L. Chen, Y. Liu, and Z. Wang, "A Survey on Generative AI for Tourism and Cultural Heritage Applications," *IEEE Access*, vol. 11, pp. 45230–45251, 2023.

[8] Gemini Team, Google, "Gemini: A Family of Highly Capable Multimodal Models," *arXiv preprint arXiv:2312.11805*, Dec. 2023.

[9] H. Patel and M. Desai, "Smart Tourism: An AI-Based Mobile Guide System Using Image Recognition and NLP," *International Journal of Computer Applications*, vol. 184, no. 21, 2022.

[10] J. Fu, H. Zheng, and T. Mei, "Look Closer to See Better: Recurrent Attention Convolutional Neural Network for Fine-Grained Image Recognition," in *Proc. IEEE CVPR*, 2017.

[11] R. Rodrigues and J. Mendes, "Heritage Site Preservation and Visitor Experience Enhancement Using AI and Augmented Reality," *Journal of Cultural Heritage*, vol. 62, pp. 144–156, 2023.

[12] S. Ioffe and C. Szegedy, "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift," in *Proc. International Conference on Machine Learning (ICML)*, Lille, France, pp. 448–456, 2015.

[13] F. Chollet, "Keras: The Python Deep Learning Library," *GitHub Repository*, https://github.com/keras-team/keras, 2015.

[14] M. Abadi et al., "TensorFlow: A System for Large-Scale Machine Learning," in *Proc. 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI)*, 2016.

[15] S. Rajpurkar, J. Zhang, K. Lopyrev, and P. Liang, "SQuAD: 100,000+ Questions for Machine Comprehension of Text," in *Proc. EMNLP*, 2016.

[16] Archaeological Survey of India, "Raigad Fort," Ministry of Culture, Government of India. [Online]. Available: https://asi.nic.in/.

[17] Maharashtra Tourism, "Raigad Fort — Maharashtra Heritage Sites," Government of Maharashtra. [Online]. Available: https://www.maharashtratourism.gov.in/.

[18] J. Sarkar, *Shivaji and His Times*, 6th ed., Orient Longman Publications, Calcutta, India, 1973.

[19] UNESCO World Heritage Centre, "Cultural Heritage Preservation Frameworks," United Nations Educational, Scientific and Cultural Organization. [Online]. Available: https://whc.unesco.org/.

[20] S. Tijtgat, W. De Waegenaere, B. Volckaert, T. Demeester, D. P. Mandal, J. Turcsany, and F. Ongenae, "Real-Time Embedded Age and Gender Classification in Unconstrained Video," in *Proc. IEEE CVPR Workshops*, 2017, pp. 1–9.

---

*— End of Paper —*
