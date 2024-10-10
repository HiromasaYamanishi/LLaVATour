# LLaVATour

## Overview

LLaVATour is a large-scale multimodal model specialized for Japanese tourism. It is trained on a dataset collected from Jalan.net, consisting of over 1.3 million entries covering various tasks such as landmark recognition, description generation, and review generation.

<div style="display: flex; justify-content: space-between;">
    <img src="images/dataset_distribution.png" width="48%" alt="Dataset Distribution">
    <img src="images/model_architecture.png" width="48%" alt="Model Architecture">
</div>

## Results

### Landmark Recognition

| Model | Acc ↑ (total) | Acc ↑ (pop) | Acc ↑ (mid) | Acc ↑ (low) |
|-------|---------------|--------------|--------------|--------------|
| ChatGPT-4V | 0.026 | 0.0771 | 0.0214 | 0.0172 |
| ChatGPT-4o | 0.155 | 0.275 | 0.160 | 0.119 |
| Gemini-Pro-1.5 | 0.092 | 0.174 | 0.093 | 0.071 |
| ResNet18-FT | 0.143 | 0.207 | 0.152 | 0.119 |
| LLaVA-1.5 | 0.0169 | 0.0378 | 0.0161 | 0.0123 |
| LLaVA-Tour | **0.166** | 0.243 | **0.192** | **0.122** |

### Review Generation

| Model | BLUE ↑ | R-1 ↑ | R-L ↑ | DIV ↑ | PROPN ↑ | TFIDF15 ↑ |
|-------|--------|-------|-------|-------|---------|-----------|
| ChatGPT-4V | 0.622 | 0.250 | 0.165 | **0.955** | 0.278 | 0.100 |
| LLaVA-Tour | **1.489** | 0.249 | **0.179** | 0.919 | **0.449** | **0.127** |
| LLaVA-Tour + context | **1.574** | **0.258** | **0.180** | 0.924 | **0.481** | **0.139** |

### Case Examples

#### Landmark Recognition

<div style="display: flex; justify-content: space-between;">
    <div style="width: 48%;">
        <h4>LLaVA-Tour</h4>
        <p>Ginzan Onsen in the Snow</p>
    </div>
    <div style="width: 48%;">
        <h4>ChatGPT-4V</h4>
        <p>Based on the uploaded image, it is evident that this is a traditional Japanese town. However, pinpointing the exact location is difficult. The red bridge visible in the photo is a characteristic feature commonly seen in historical towns such as Gion in Kyoto, Takayama in Gifu, and Kinosaki Onsen in Hyogo.</p>
    </div>
</div>

![Landmark Recognition Example](images/landmark_recognition_example.png)

#### Review Generation

<div style="display: flex; justify-content: space-between;">
    <div style="width: 48%;">
        <h4>LLaVA-Tour</h4>
        <p>This is the place where the famous Clark statue is located. You can get a panoramic view of Sapporo, so I recommend visiting on a clear day. If you come to Sapporo, you should visit this place at least once.</p>
    </div>
    <div style="width: 48%;">
        <h4>LLaVA-1.5</h4>
        <p>The statue of the person stands on top of the hill.</p>
    </div>
</div>

![Review Generation Example](images/review_generation_example.png)
