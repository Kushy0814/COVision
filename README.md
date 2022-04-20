![](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/HTML-239120?style=for-the-badge&logo=html5&logoColor=white)
![](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![](https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)
![](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![](https://img.shields.io/badge/Maintained%3F-yes-green.svgr)
![](https://img.shields.io/website-up-down-green-red/http/monip.org.svg)
![](https://img.shields.io/badge/-Google%20Colab-blue)

# COVision: A Novel CNN for the Differentiation of COVID-19 From Common Pulmonary Conditions (ROBO052T)

With the growing amount of COVID-19 cases, especially in developing countries with limited medical resources, it’s essential to accurately diagnose COVID-19 with high specificity. Due to characteristic ground-glass opacities (GGOs) and other features present in both COVID-19 and other acute lung diseases, misdiagnosis occurs often in manual interpretations of CT scans. Current deep learning models can identify COVID-19 but *cannot distinguish* it from other common pulmonary diseases like bacterial pneumonia. COVision is a novel multi-classification convolutional neural network (CNN) that can differentiate COVID-19 from other common pulmonary diseases, with a high specificity.

## Usage
COVision is integrated into a [website](https://covision.timmy625.repl.co/) where the user can upload a CT volume for a patient and then input the patient’s clinical factors (age, cough characteristics, etc). Once the data is inputted, COVision outputs the probabilities of the patient having COVID-19 (viral pneumonia), bacterial pneumonia, or is healthy. A summary of the diagnosis and a summary of clinical factors can be downloaded as a .txt file making results easily shareable. 

![Web capture_15-4-2022_123523_covision timmy625 repl co](https://user-images.githubusercontent.com/30708141/163596786-4c101603-2d86-41ee-84c9-c877c115c886.jpeg)

## Built With
- [Python in Google Colaboratory](https://colab.research.google.com/)
- [Jupyter Notebook](https://jupyter.org/)
- [HTML/CSS/JS](https://developer.mozilla.org/en-US/docs/Web/HTML)

## Data Availability
CT Scans of COVID-19, pneumonia, and healthy patients were obtained from the [China Consortium of Chest CT Image Investigation (CC-CCII) dataset](http://ncov-ai.big.ac.cn/download?lang=en). Ground truth for the CC-CCII dataset was established via serology tests and confirmed by laboratory findings. Clinical factors for COVID-19, and pneumonia patients were obtained from the [Khorshid COVID Cohort (KCC) study](https://figshare.com/articles/dataset/COVID-19_and_non-COVID-19_pneumonia_Dataset/16682422). Clinical factors for healthy patients were obtained from the [Israeli Ministry of Health](https://data.gov.il/dataset/covid-19/resource/74216e15-f740-4709-adb7-a6fb0955a048). Ground truth for clincials factors was also established via laboratory findings and serology tests.
