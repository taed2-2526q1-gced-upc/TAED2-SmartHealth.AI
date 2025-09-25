
# Dataset Card for Estimation of Obesity Levels Based on Eating Habits and Physical Condition

<!-- Provide a quick summary of the dataset. -->

This dataset contains survey data collected from individuals in Mexico, Peru, and Colombia to estimate obesity levels based on eating habits and physical condition. It includes 2,111 rows and 17 columns with demographic, dietary, and lifestyle features, along with a categorical target variable for obesity levels.

## Dataset Details

### Dataset Description

This dataset was introduced by **Palechor & de la Hoz Manotas (2019)** and published in *Data in Brief*.  
It contains information collected from individuals in **Colombia, Peru, and Mexico** (ages 14–61) to estimate obesity levels based on eating habits and physical condition.

- **Curated by:** Fabio Mendoza Palechor & Alexis de la Hoz Manotas (Universidad de la Costa, Colombia)  
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)  
- **Size:** 2,111 records, 17 attributes  
- **Shared by:** UCI Machine Learning Repository
- **Language(s):** Original survey in Spanish; released dataset provided in English.
- **Labels:** NObeyesdad (multi-class)  
  - Insufficient Weight  
  - Normal Weight  
  - Overweight Level I  
  - Overweight Level II  
  - Obesity Type I  
  - Obesity Type II  
  - Obesity Type III  

### Dataset Sources

- **Repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)  
- **Paper:** [Palechor & de la Hoz Manotas, 2019](https://doi.org/10.1016/j.dib.2019.104344)  

## Uses

### Direct Use

- Educational purposes (e.g., coursework in machine learning or MLOps such as TAED2 at UPC).  
- Research on obesity prediction and prevention using lifestyle and physical condition indicators.   

### Out-of-Scope Use

- **Not suitable for clinical use, diagnosis, or medical decision-making.**  
- Should not be used in production healthcare systems.  
- Cannot substitute for professionally collected, diverse, and validated health data.  
- Using the dataset to stigmatize or discriminate individuals.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

- **Rows:** 2,111
- **Columns:** 17
- **Target Variable:** NObeyesdad (7 obesity categories)

**Class Distribution:**

- Obesity_Type_I: 351
- Obesity_Type_II: 297
- Obesity_Type_III: 324
- Overweight_Level_I: 290
- Overweight_Level_II: 290
- Normal_Weight: 287
- Insufficient_Weight: 272

- **Features include:**  
  - Demographics: Gender, Age  
  - Anthropometrics: Height, Weight  
  - Eating habits: High-caloric food (FAVC), Vegetables (FCVC), Main meals (NCP), Snacks (CAEC), Water intake (CH2O), Alcohol consumption (CALC)  
  - Lifestyle/physical condition: Smoking (SMOKE), Calories monitoring (SCC), Physical activity (FAF), Time using technology (TUE), Transport method (MTRANS)  

- **Data types:** Mixed (categorical and continuous).  
- **Splits:** No predefined train/test split.  


## Dataset Creation

### Curation Rationale

The dataset was created to support research on **intelligent computational tools** for obesity estimation and to facilitate the development of recommender systems that promote healthier lifestyles.  

### Source Data

#### Data Collection and Processing

- Collected via an **online survey** (30 days, anonymous participation).  
- Initial dataset: 485 records.  
- Preprocessing included removal of missing/atypical data and normalization.  
- Labels assigned using BMI thresholds (WHO and Mexican normativity).  
- To correct class imbalance, **SMOTE oversampling in Weka** generated additional synthetic records.  
- Final dataset: 2,111 records, with balanced distribution across classes.  


#### Who are the source data producers?

Anonymous individuals from **Mexico, Peru, and Colombia** who completed the online survey.

#### Personal and Sensitive Information

The dataset does not include personally identifiable information.  
It does, however, contain **sensitive health-related data** (weight, eating habits, activity, etc.), so it must be treated responsibly.

### Annotations

#### Annotation process

Labels were computed using BMI thresholds following WHO and Mexican normativity.

## Bias, Risks, and Limitations

- Data is self-reported (possible inaccuracies).
- The dataset represents individuals from only three Latin American countries → limited geographic and cultural diversity.  
- **77% synthetic data** may introduce distributional artifacts not present in real populations.  
- Lifestyle-based features ignore **genetic, environmental, and socioeconomic factors** influencing obesity.  
- Models trained on this dataset may oversimplify complex health conditions.  

### Recommendations

- Use primarily for **educational and research** purposes.  
- Do not generalize results to global populations.  
- For clinical or health policy applications, supplement with **real, representative, and validated data**.  

## Citation

If you use this dataset, please cite the original publication and reference the UCI repository:

**APA:**  
Palechor, F. M., & de la Hoz Manotas, A. (2019). *Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico*. Data in Brief, 25, 104344. https://doi.org/10.1016/j.dib.2019.104344  

Available at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition).  

**BibTeX:**  
```bibtex
@article{palechor2019dataset,
  title={Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico},
  author={Palechor, Fabio Mendoza and de la Hoz Manotas, Alexis},
  journal={Data in Brief},
  volume={25},
  pages={104344},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.dib.2019.104344},
  url={https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition}
}