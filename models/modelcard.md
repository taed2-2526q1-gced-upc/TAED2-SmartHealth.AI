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


### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

Use only for coursework and experimentation.  
For any health-related applications, real, representative, and clinically validated datasets are required.

## Model Card Authors

Steffen, Renaux, Matilde (SmartHealth-AI student project group)  