# Model Card for SmartHealth-AI (Obesity Guidance Tool)

<!-- Provide a quick summary of what the model is/does. -->

A Random Forest model that uses health-related features to guide users toward healthier habits in relation to obesity.

## Model Details

### Model Description

This is a **Random Forest classifier** trained to estimate obesity levels based on lifestyle and anthropometric features.  
The model was developed as part of the **TAED2 course (Universitat Politècnica de Catalunya, FIB)** and is intended **solely for educational purposes**.  
It demonstrates preprocessing, hyperparameter tuning, evaluation, and reporting practices for ML systems.

- **Developed by:** Steffen, Renaux and Matilde  
- **Model type:** Supervised learning – Random Forest
- **Language(s) (NLP):** English 
- **License:** MIT

### Model Sources

- **Repository:** https://github.com/taed2-2526q1-gced-upc/TAED2-SmartHealth.AI
- **Dataset card:** [Estimation of Obesity Levels dataset](./DATACARD.md)  

## Uses

### Direct Use

The model is intended to be used in a **SmartHealth-AI application** that provides users with lifestyle guidance when obesity risk is detected.  
Its purpose is to illustrate how machine learning can be applied to health-related data for preventive insights.  

**Note:** This implementation was developed as part of the **TAED2 course (UPC)** and is **for educational purposes only**.  
The output must not be interpreted as medical advice, but as an example of lifestyle guidance in an academic setting.

### Downstream Use

It could be integrated into digital health platforms or apps that promote preventative health and wellness.

### Out-of-Scope Use

- Should not be used as a replacement for professional medical advice, diagnosis, or treatment.  
- Not intended for critical medical decision-making or emergency situations.  

## Bias, Risks, and Limitations

The model relies on correlations between lifestyle features and obesity but may not capture all medical, genetic, or environmental factors.  
Bias may occur if the dataset does not represent diverse populations (e.g., age groups, socioeconomic backgrounds, geographic regions).  
Since the dataset includes synthetic components, the results may not generalize to real-world populations.  

Recommendations derived from the model may oversimplify complex health conditions.  
If misused in a medical context, the model could lead to inappropriate lifestyle guidance or false reassurance.

## Carbon Footprint

**Training Emissions:**
- **Total CO2 Emissions:** 0.241 g CO2eq (0.000241 kg CO2eq)
- **Total Energy Consumed:** 0.00138 kWh
- **Training Duration:** 16.15 seconds (0.0045 hours)
- **Source:** Measured using CodeCarbon v3.0.7
- **Training Type:** Fine-tuning
- **Hardware Used:** 
  - CPU: AMD Ryzen 7 7435HS (16 cores)
  - RAM: 15.69 GB
- **Geographic Location:** Barclona, Spain
- **Carbon Intensity:** ~174 gCO2eq/kWh (Spain regional grid)
- **Cloud Provider:** N/A (local machine training)
- **PUE:** 1.0 (Power Usage Effectiveness)

**Environmental Context:**
This training footprint is equivalent to approximately 0.001% of the emissions from a single kilometer driven by an average passenger vehicle, or about 0.000014% of the average American's daily carbon footprint.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

Use only for coursework and experimentation.  
For any health-related applications, real, representative, and clinically validated datasets are required.

## Training Details

### Training Data

The model was trained on the [Estimation of Obesity Levels dataset](./DATACARD.md), which contains lifestyle and anthropometric features related to obesity classification.

### Training Procedure

#### Preprocessing

Standard preprocessing steps were applied including:
- Feature scaling/normalization
- Handling of categorical variables
- Train/validation split

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The model was evaluated on a held-out validation set from the obesity dataset.

#### Metrics

- **Accuracy:** 97.16%
- **F1 Score (Macro):** 97.09%

### Results

The model achieves high performance on the validation set:

| Metric | Value |
|--------|-------|
| Validation Accuracy | 0.972 |
| Validation F1 Macro | 0.971 |

## Model Card Authors

Steffen, Renaux, Matilde (SmartHealth-AI student project group)  