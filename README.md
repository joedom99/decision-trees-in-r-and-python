# decision-trees-in-r-and-python
# Smarter Marketing with Decision Trees: A Practical Introduction Using R and Python

## Overview
This repository contains R and Python implementations of a **decision tree model** designed to predict whether an online shopper will make a purchase. The analysis is based on the **Online Shoppers Purchasing Intention Dataset**, which includes user behavior data from an e-commerce website. The goal is to explore how decision trees can help marketers understand customer intent and optimize digital strategies.

Both implementations train a **classification tree** using key behavioral metrics such as **PageValues, BounceRates, and ExitRates** to distinguish between purchasers and non-purchasers. While the **R version** works efficiently with minimal tuning, the **Python version** requires additional adjustments for feature encoding, pruning, and class balancing.

For more information and a complete code walk-through, please check out my blog at https://blog.marketingdatascience.ai.

## Features
- **R Implementation:** Uses the `rpart` package to build and visualize a decision tree with simple tuning.
- **Python Implementation:** Uses `scikit-learn` to construct a decision tree, with more extensive parameter adjustments for improved balance.
- **Data Processing:** Includes preprocessing steps like categorical encoding and train-test splitting.
- **Evaluation Metrics:** Generates a **confusion matrix, classification report, and feature importance scores** for model assessment.
- **Visualization:** Decision trees are plotted for both R and Python, showing key decision points.

## Dataset
This project is based on the **Online Shoppers Purchasing Intention Dataset**, which contains **12,330 sessions** collected from an e-commerce website. The dataset includes **17 behavioral features** such as time spent on different types of pages, bounce rates, and traffic source.

**Citation:**  
Sakar, C. O., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository.  
[https://doi.org/10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)

## License
This project is open-source under the MIT License.
