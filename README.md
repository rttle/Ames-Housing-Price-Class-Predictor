<img width="200" alt="image" src="https://github.com/rttle/Bank-Churn-Kaggle-Challenge/assets/143844181/dbbeb760-7ac3-4d53-84ce-a08071725da1">

# Predicting Housing Sales Price Categories in Ames, Iowa
This repository holds an attempt to apply machine learning on housing data from Ames, Iowa to predict the sales price category, the data used is provided by the Ames Assessor’s Office for tax assessment purposes and shared through Kaggle: https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset/data.

## Overview
This project takes the housing data originating from the Ames, Iowa Tax Assessor office to train a model to predict housing price categories for the purpose of mortgage auction pricing. Categories, price ranges, are specifically being looked at as a way to appeal to customers fearing the worst situation and trying to maintain equity. The use of price ranges would benefit the customers in that regard, while also continuing to provide the efficiency and loss protection found in the traditional approach of mortgage auctions compared to mortgage sales.
This repository shows how this project was approach and how the final model evolved to create a price category predictor. From the metrics referenced (accuracy, precision (macro), recall (macro), and F1 score (macro)), the XGBoost model with oversampling to address imbalance performed the best with a f1 macro score of 0.81. This was an increase of ~0.10 in comparison to the updated baseline, which had a f1 macro score of 0.70.

## Summary of Workdone
### Data
- Data: Type: Tabular
  - Input: CSV file of features, output: Price Bins (created from Sales Price column, a continuous numerical feature in original CSV file).
- Size: 2930 observations, 82 variables
- Instances (Train, Test, Validation Split): 
  - Train: 80%
  - Test: 20%
  - Validation: Cross-Validation on Train set
 
### Preprocessing / Clean up
**Missing Values/Duplicates.** Missing values were dealt with using either logically understanding of the data (Example: house has no fireplaces, so there cannot be any quality information) or using median value (by neighborhood) to fill missing values with understanding that such use narrows the distribution of the feature.

**Dropped.** All ID columns were dropped from the dataset. Repetitive or highly correlated columns were also dropped.

**Outliers.** Outliers were attempted to be address using normalization; however, the final algorithm chosen, XGBoost, had the same performance whether outliers were dealt with using normalization or relying on the inherent abilities of the algorithm.

**Feature Engineering.** Feature engineering was done to aggregate square feet/area related features to address the high dimensionality found in the dataset. Feature engineering was also done to create features that would help indicate high value houses, helping with predicting the highest class.

**Encoding.** One Hot and Ordinal Encoding had to be done in preparation for training the machine learning algorithms. One Hot Encoding was done to all nominal, categorical columns. Ordinal Encoding was applied to all ordinal, categorical columns.

### Data Visualization
The figure below is a summary table of the dataset. Note that the categorical/numerical classification is initial determinations and changed as a better understanding of the dataset was reached. These initial determinations were made by setting definitions through use of functions.
 
<img width="468" height="371" alt="image" src="https://github.com/user-attachments/assets/dd2e464f-51db-43b4-9b7e-70030cbeb19a" />


<img width="468" height="401" alt="image" src="https://github.com/user-attachments/assets/16750c3f-132c-4679-bae3-93976e08e1d1" />


<img width="468" height="252" alt="image" src="https://github.com/user-attachments/assets/ddc8fa5d-d80a-40fb-a155-84f42900bbfb" />


<img width="468" height="354" alt="image" src="https://github.com/user-attachments/assets/42447bd6-478e-46be-b943-fc70c94b3264" />


The figure below is a portion of a pairplot, which was meant to show relationships of the features through the scatterplots and the distribution of the target variable classes, Price Bins. The pairplot also easily showed what should be categorical features when bars of data were shown in the scatterplots. 
 
<img width="529" height="490" alt="image" src="https://github.com/user-attachments/assets/d48b454d-fb31-4e76-87b4-51a94b8c8e11" />


Below is a bar graph that shows the dataset target classes are imbalanced.

<img width="280" height="257" alt="image" src="https://github.com/user-attachments/assets/bae478fa-bee1-44f5-baf4-7d8b4e5716f4" />


### Problem Formulation
- Input / Output
  - Input: 34 numerical features, 46 categorical features
  - Output: Price Bins
- Models 
  - Decision Tree
  - Random Forest
  - K-Nearest Neighbors
  - XGBoost
  - CatBoost
  - AdaBoost
- Hyperparameters
  - The following were the hyperparameters used for XGBoost. 
<img width="377" height="188" alt="image" src="https://github.com/user-attachments/assets/64dce9a2-43f4-4c3e-972d-0f14299ff8b5" />


### Training
For the baseline models, the dataset was split 70% for training and 30% for testing. Multiple models were trained by using dictionaries of classifiers.
As improvement to the model became the focus, the train/test split was switch to 80/20 in consideration for the minimal number of observations, ~3000. To address the class imbalance, stratification and oversampling was implemented. SMOTE was also tested, but it was found that oversampling performed better.
Grid Search was used to perform hyperparameter tuning and perform cross validation. Hyperparameter tuning was switched to RandomizedSearch to optimize time management.

### Performance Comparison
Multiple metrics were computed for the models, including: accuracy, precision, recall, and F1 score. All were included in a comparison table; however, of importance is the F1 score. Confusion matrix was also used to visualize model performance because it reflects the scores seen in the table. Classification reports were also generated for models of interest to further understand how well the model was performing.
Below is table of metrics for the models trained. From the table, Decision Tree performed the best across the board.

<img width="333" height="361" alt="image" src="https://github.com/user-attachments/assets/c9ba439e-34c1-461b-a6e2-51c4c9870ffb" />


The finalized model was the XGBoost model, making use of the full set of features with oversampling applied to address the imbalance. This model was trained using the 80/20 stratified split. The macro average scores seen in the table above, showcase how the model’s performance was balanced. This is further seen in the generated classification report.
Below is the Confusion Matrix for the final model.

<img width="439" height="384" alt="image" src="https://github.com/user-attachments/assets/07949bb4-0cef-40e4-bf25-b37826c94060" />


Below is the classification report for the final model.

<img width="407" height="213" alt="image" src="https://github.com/user-attachments/assets/80373d02-5f71-4051-a6f3-2ea4505a51fa" />


### Conclusions
Of the models trained, XGBoost, trained using the full set of features and oversampling to address imbalance, did the best at predicting the housing sales price category.  

### Future Work
To achieve better results, moving to a time series to allow the model to use past information to predict house sale price categories for the current time. This would address a limitation of the current final model, which is trained on data between 2006 and 2010. Additionally, it was noted that the removal of Sales Type would benefit the model. These two implementations are simple changes that can further this project and open the door to more better model performance through further feature engineering.

## How to reproduce results
To reproduce results, download the csv file from the linked Kaggle page. Then ensure that the Housing_Preprocessing.py file is downloaded from this repository and run the 4_XGBoost_Modeling.ipynb notebook also found in this repository.

## Overview of files in repository
- **1_DataUnderstanding+Baseline.ipynb:** Notebook that takes the dataset to create tables and visualizations for data understanding. It also creates the initial baseline and updated baseline that was used for the rest of the project.
- **2_More_DataUnderstanding+Normalization.ipynb:** Notebook that involves some more data understanding, along with some data preprocessing (specifically normalization methods). 
- **3_Initial_XGBoost_Model.ipynb:** Notebook showing additional data understanding, along with the use of a stronger model, XGBoost, in comparison to the updated baseline.
- **4_XGBoost_Modeling.ipynb:** Notebook showing the further development of the XGBoost model trained in the previous notebook. This notebook houses the final model.
- **5_CatBoost+AdaBoost_Modeling.ipynb:** Notebook showing the testing of other algorithms to determine the best train model for this project’s purposes. This notebook also houses some testing with feature engineering that creates features meant to indicate higher value houses.
- **Housing_Preprocessing.py:** Module created to wrap all preprocessing done to the dataset . 

## Data
Data is from the Ames, Iowa Tax Assessor Office; with the dataset being shared through Kaggle. https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset/data

## Citations
-Thapa, S., *Ames Housing Dataset *, WWW.KAGGLE.COM, 2023, https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset/data.
