# AI_Projects_and_Experiments
My personal projects, experiments and school practices about AI (general AI course, ML, and DL).

## Description
This repository contains artificial intelligence projects and experiments that I developed in my university courses. Also includes some personal projects and experiments that I developed on my own. Due to my university courses were taken in Spanish, some of the notebooks and scripts were written in Spanish. So feel free to ask any questions.

## Author and Contact
Arnoldo Fernando Chue Sánchez: arnoldwork20@gmail.com

## List of Projects 

### Deep Learning: Decrappify. Super Resolution Model
[View Project](DL_Decrappify/Decrappify.ipynb)
- Explored decrappify models by utilizing U-Net architectures to reconstruct degraded images with high fidelity.
- Implemented a super-resolution model capable of transforming low-resolution images (32x32 pixels) into high-resolution outputs (256x256 pixels), effectively reconstructing images from just 1/64th of their original resolution.

### Deep Learning: Segmentation with Customized Metrics
[View Project](DL_Segmentation/Segmentation.ipynb)
- Conducted experiments with segmentation models to detect multiple objects within a single image.
- Designed and implemented custom performance metrics to evaluate model accuracy for a specific category, leveraging advanced tensor manipulations in PyTorch.

### Deep Learning: Tabular Data. Combination of continuous and categorical variables with Pytorch and Fastai
[View Project](DL_Tabular_Data/TabularData.ipynb)
- Data exploration and cleaning of Higgs dataset
- Implementation of a neural network for tabular data using categorical and continuous variables

### Deep Learning: Simultaneous Classification and Regression for Images
[View Project](DL_multiple_classification_and_regression/Multiple_classification_regression.ipynb)
- Developed a convolutional neural network (CNN) capable of predicting multiple categories (classification) and a continuous value (regression) from a single image.
- Designed and implemented custom loss functions and metrics to handle the dual-task learning effectively.
- Performed advanced tensor manipulations to optimize model outputs and improve computational efficiency.

### Deep Learning: Cartoon ImageNet Sample Classification Contest
[View Project](DL_CartoonImageNet_Classification)
- Achieved first place in a deep learning model competition for classification using a sample of ImageNet with cartoon-style transformations.
- Worked under strict constraints:
    - Fewer than 10 million parameters.
    - Training limited to 12 epochs.
    - Images capped at 256x256 resolution.
    - Maximum training time of 2 minutes per epoch on a system with 11 GB of VRAM.
- Designed and implemented custom layers, combining convolutions, normalization, and diverse activation functions.
- Developed and optimized ResBlocks with innovative modifications to enhance model performance.
- Entire project was coded using PyTorch and FastAI frameworks.

### Deep Learning: Animals Classification Using Resnet Architecture
[View Project](DL_Animals_Clasifications_with_Resnet/Animals_Arnold.ipynb)
- Conducted an animal classification task leveraging PyTorch and FastAI frameworks.
- Designed and implemented custom data augmentation and image transformation techniques to enhance model performance.
- Utilized the ResNet-18 architecture from Torch for feature extraction and classification.

### Deep Learning Certificate: Large Language Model Development
[View Project](DL_Certification/DL_LLM_prompt_experiments/Tarea6_Arnoldo.ipynb)
- Connected to Llama Studio to leverage a pretrained Large Language Model (LLM).
- Designed and optimized prompts to develop an LLM for vocational guidance, ready for production deployment at a university.

### Deep Learning Certificate: Introduction to Transformer Architecture 
[View Project](DL_Certification/DL_transformer_translate/Tarea5_Arnoldo.ipynb)
- Processed and prepared a dataset to build a Hebrew-Spanish translation corpus.
- Implemented a compact transformer architecture for translating text from Spanish to Hebrew.

### Deep Learning Certificate: Recurrent Neural Networks for Sentiment Analysis
[View Project](DL_Certification/DL_RNN_sentiment_analysis/Tarea4_Arnoldo.ipynb)
- Implemented sentiment analysis on tweets using Recurrent Neural Networks (RNNs).
- Explored the fundamentals of RNN architectures through hands-on experiments with TensorFlow and Keras.
- Note: While modern architectures like transformers offer superior performance, this project focused on understanding and applying RNNs as a foundational learning exercise.

### Deep Learning Certificate: Transfer Learning for Fresh Fruit Detection
[View Project](DL_Certification/DL_transfer_learning_fruit/Tarea3_Arnoldo.ipynb)
- Utilized transfer learning with a pretrained model to achieve high accuracy in detecting fresh and rotten fruit.
- Gained hands-on experience with image preprocessing and manipulation using TensorFlow and Keras.

### Deep Learning Certificate: Convolutional Neural Network Models for Mosquito Species Detection
[View Project](DL_Certification/DL_mosquito/Tarea2_Arnoldo.ipynb)
- Preprocessed mosquito image datasets to prepare high-quality input for training convolutional neural networks.
- Trained and evaluated multiple convolutional neural network (CNN) architectures in Tensorflow and Keras to enhance detection accuracy for dangerous mosquito species.

### Deep Learning Certificate: First Aproach to Convolutional Neural Networks
[view Project](DL_Certification/DL_CNN_first_approach/Tarea1_Arnoldo.ipynb)
- Implementation of different CNN architectures in tensorflow and keras for image classification

### Machine Learning: Supervised Learning Final Project  
[View Project](ML_Supervised_Learning_ML_Class_Project)  
- Implemented traditional machine learning algorithms and combined them into an **ensemble model**.  
- Compared ensemble learning methods using diverse evaluation metrics.  
- Conducted hyperparameter tuning for each component of bagging ensembles to optimize performance.  

### Machine Learning Metrics and Cross-Validation  
[View Project](ML_Metrics_And_Cross_Validation/Proyecto.ipynb)  
- Deployed multiple classification algorithms, including **decision trees**, **k-NN**, **Naive Bayes**, **logistic regression**, and **support vector machines**.  
- Optimized models to maximize key metrics such as **accuracy**, **recall**, **precision**, and **F1-score**.  
- Utilized cross-validation to prevent overfitting and improve model generalization.  
- Conducted metric-driven decision-making to select the best model for specific use cases.  

### Machine Learning Certificate: Unsupervised Learning Project  
[View Project](ML_Certification/ML_Certificate_Practice_KMeans_KMedoids/Tarea4_ArnoldoFernandoChueSánchez.ipynb)  
- Applied **K-Means** and **K-Medoids** clustering algorithms for client segmentation.  

### Machine Learning Certificate: Logistic Regression Project  
[View Project](ML_Certification/ML_Certificate_Practice_Logistic_Regresion)  
- Developed a logistic regression model to classify Uber and Lyft users in Boston.  

### Machine Learning Certificate: k-Nearest Neighbors Project  
[View Project](ML_Certification/ML_Certificate_Practice_KNN/Tarea3_ArnoldoFernandoChueSánchez.ipynb)  
- Implemented the **k-Nearest Neighbors (k-NN)** algorithm to classify water deposits using real data from the state government.  

### Machine Learning Certificate: Decision Tree Project  
[View Project](ML_Certification/ML_Certificate_Practice_Decision_Tree_Clasification/Tarea1_ArnoldoFernandoChueSánchez.ipynb)  
- Built a decision tree model to classify social media platform users into distinct age groups.  

### 15 Puzzle Solver  
[View Project](15_puzzle/15_puzzle_classic.ipynb)  
- Utilized the **A\*** algorithm with traditional heuristics to solve the 15-puzzle problem efficiently.  


## License
This project is licensed under the GNU General Public License v3.0.

Feel free to make any necessary changes or suggestions.