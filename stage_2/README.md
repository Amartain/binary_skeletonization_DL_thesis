# Stage 2 

# 0. Data Pipeline
- Choose 2 additional datasets
    have already: Kimia 99, Kimia 216
    NEXT: Animal2000, MPEG400

# 1. Models
- Unet from stage 1
- Res UNet
- Attention UNet
- "2 stage UNet" - rough skeleton >>> refine skeleton ensure 1px wide 

# 2. Losses
- DiceLoss w/ BCE then categorical cross entropy???
- Weighted Categorical Cross Entropy
- Weighted Focal Loss
# 3. Training

# 4. Evaluation
##  4.1 Metrics
- F1
- Relaxed / Slack F1
- Average Error Pixel AEP
- Average Thickness and 99th max thickness
- Centerline Dice clDice



# -. TESTs / func / file / module