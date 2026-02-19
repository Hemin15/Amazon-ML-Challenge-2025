# ML Challenge 2025: Smart Product Pricing Solution

**Team Name:** Tech Titans  
**Team Members:** Hemin Modi, Rahul Shah, Rudra Patel
**Submission Date:** 13-10-2025

## 1\. Executive Summary

Our approach combines multimodal feature extraction and ensemble learning to predict product prices accurately. We engineered diverse features — numeric attributes, regex-based pack/weight/volume extraction, and TF-IDF text embeddings — and integrated them with CLIP image embeddings for visual understanding. Finally, we used an ensemble of LightGBM and CatBoost regressors with stratified cross-validation and SMAPE optimization, achieving robust performance and interpretable feature importance.



## 2\. Methodology Overview

### 2.1 Problem Analysis

The task involves predicting the accurate price of products using multimodal data — including text descriptions, images, and structured attributes. This is challenging because product listings often contain noisy, incomplete, or inconsistent information, such as varied units, ambiguous titles, and missing images. Additionally, textual and visual data contribute differently to price prediction, requiring an effective fusion of heterogeneous features. Hence, the core problem lies in designing a robust pipeline that can extract meaningful representations from both text and images, handle data imbalance and missing values, and build a generalizable regression model capable of capturing real-world pricing patterns.

**Key Observations:**

### 2.2 Solution Strategy

Our solution adopts a multimodal learning approach that integrates both textual and visual information to enhance price prediction accuracy. Textual features are extracted from product descriptions using NLP embeddings, while visual features are obtained from product images using the CLIP (Contrastive Language–Image Pretraining) model. These multimodal features are then combined and fed into a regression model, enabling a deeper understanding of both content and visual context for precise price estimation.



Approach Type: Hybrid (Multimodal Deep Learning)
Core Innovation: Our main technical contribution is the development of a multimodal hybrid framework that effectively combines visual and textual information for product price prediction. We utilize CLIP (Contrastive Language–Image Pretraining) to extract semantically aligned embeddings, where the text encoder processes product descriptions and the vision transformer (ViT) extracts image features. Both sets of embeddings are projected into a shared latent space, ensuring that visual and textual semantics are aligned. These features are then concatenated to form a unified multimodal representation, which is passed through a regression model to predict product prices. By leveraging transfer learning from the pretrained CLIP model, our approach enhances learning efficiency and accuracy even with limited data. The model captures cross-modal relationships between product appearance and textual attributes, enabling better generalization. Overall, our innovation lies in the feature-level fusion of image and text embeddings to achieve context-aware and robust price estimation.



## 3\. Model Architecture

Text Processing Pipeline:

Preprocessing steps: Lowercasing, punctuation removal, tokenization, stop word removal, and conversion to embeddings using CLIP text encoder.
Model type: CLIP Text Encoder (Transformer-based model pretrained on image-text pairs).
Key parameters: Embedding dimension = 512, context length = 77 tokens, pretrained weights from Open AI CLIP.

Image Processing Pipeline:
Preprocessing Steps: Resize images to 224×224, normalize using CLIP’s mean/std, convert to tensor, optional data augmentation.
Model Type: CLIP Image Encoder (Vision Transformer or ResNet backbone).
Key Parameters: 512-dim embeddings, input size 224×224, pretrained CLIP weights, normalized embeddings for cosine similarity.



## 4\. Model Performance



4.1 Validation Results



SMAPE Score: 56.41% (ensemble, uncalibrated)



SMAPE Score (after isotonic calibration): 62.06%



Other Metrics: Not explicitly calculated, but the ensemble consistently outperformed individual models (LightGBM: 56.81%, CatBoost: 57.40%, Ridge: 63.80%)



Observations:



LightGBM contributed the most to the final ensemble (coarse weight 1.0).



Ridge performed the worst individually but helped stabilize predictions in the ensemble.



Isotonic calibration improved alignment of predicted prices but slightly increased SMAPE due to the OOF-to-test distribution shift.



5\. Conclusion



We built a multimodal regression pipeline combining numeric, text, and image embeddings. Using a stacked ensemble of LightGBM, CatBoost, and Ridge, we achieved robust validation performance. Key takeaways include the importance of multimodal features and ensemble weighting, as well as the effect of calibration on prediction reliability.



Appendix

A. Code artefacts



Full project code, including feature extraction and training scripts, is available here: \[Insert your drive link]



B. Additional Results



Feature importance (LightGBM \& CatBoost) highlights the most predictive features from numeric, text, and CLIP embeddings.



OOF vs. true price scatter plot shows model performance distribution across the training set.



Histograms of predicted test prices confirm the ensemble produces reasonable value ranges for all 75,000 items.

https://drive.google.com/file/d/1BLYWlWtfsDHNpt6UZo\_hxOHX7csC9e8F/view?usp=drive\_link

