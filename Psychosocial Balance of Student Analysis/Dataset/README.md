# Psychosocial Dimensions of Student Life Dataset

## Dataset Overview

This dataset compiles survey results from 100 computer science students, with the objective of uncovering correlations between depression levels, academic performance, and ADHD patterns through comprehensive data analysis.

### Dataset Column Descriptions

- **Age:** Represents the age of individuals in the dataset, providing insights into the age distribution of the study.
- **Gender:** Indicates the gender of each individual, allowing for the exploration of gender-related patterns and trends within the dataset.
- **Academic Performance:** Reflects the academic achievements of individuals.
- **Taking Note In Class:** Describes whether individuals take notes during class, providing insights into study habits and engagement during lectures.
- **Depression Status:** Indicates the presence or absence of depressive symptoms, contributing valuable information about the mental health of individuals in the dataset.
- **Face Challenges To Complete Academic Task:** Explores whether individuals encounter challenges in completing academic tasks.
- **Like Presentation:** Reflects individuals' preferences for presentations, offering insights into their learning style and engagement with visual or oral communication. This also measures if they are extroverted or introverted.
- **Sleep Per Day Hours:** Represents the average hours of sleep individuals get per day, providing information on sleep patterns and potential correlations with academic performance.
- **Number Of Friend:** Quantifies the social aspect by indicating the number of friends each individual has, contributing to the understanding of social dynamics within the dataset.
- **Like New Things:** Explores individuals' receptiveness to new experiences or concepts, offering insights into their adaptability and openness to innovation.


## Kaggle API Usage Guide

For convenient access to the dataset, you can use the Kaggle API. If you are using Google Colab or any other environment that supports Kaggle API, follow the steps below:

1. **Upload Kaggle API Key:**
   - Obtain your Kaggle API key from your Kaggle account settings and upload it to your Colab environment.

2. **Install Kaggle Python Package:**
   - Install the Kaggle package using the command: `!pip install kaggle`

3. **Download and Unzip the Dataset:**
   - Download the SUIM dataset using the Kaggle API: `!kaggle datasets download -d mdismielhossenabir/psychosocial-dimensions-of-student-life`
   - Unzip the downloaded dataset: `!unzip psychosocial-dimensions-of-student-life.zip`

## Usage in Other Environments:

If you are not using Google Colab, you can manually download the dataset from the [Kaggle website](https://www.kaggle.com/datasets/mdismielhossenabir/psychosocial-dimensions-of-student-life). After downloading the dataset, unzip it and use it in your code.
The dataset file is available in the this directory as `CSE_student_performances.csv`.

