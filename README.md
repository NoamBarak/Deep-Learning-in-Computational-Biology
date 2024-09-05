# Predicting RNAcompete Binding from HTR-SELEX Data 🧬
---  
**Introduction** 🔍
-
RNA-binding proteins (RBPs) are crucial for post-transcriptional regulation, affecting RNA splicing, stability, and translation.   
Mapping RBP-RNA interactions is essential for understanding gene regulatory networks.   
RNAcompete assays measure RBP binding affinities, while HTR-SELEX explores RNA structure through iterative selection.  

**The Project**📝  
-
The project aims to predict RNA binding intensity for each RNAcompete sequence based on HTR-SELEX datasets for various RBPs.     
The **input** consists of 1-4 sequence files per RBP, representing pools of oligonucleotides surviving after several selection cycles.     
The **output** is a predicted binding score for each sequence.    


**Model Architecture**📊  
-
***Neural Network***    
DNA sequences are one-hot encoded and processed through a Conv1D layer with 512 filters and a kernel size of 8, followed by MaxPooling and three Dense layers (63, 32, 32 units) with ReLU activation.   
A final Dense layer with softmax activation outputs class probabilities.   
The model is trained with the Adam optimizer.  

![image](https://github.com/user-attachments/assets/eb4c1745-1257-4c3d-bff7-cf46f76cc38b)

***XGboost Algorithm***  
An XGBoost model is trained using the probability vectors from the neural network to predict binding scores.

***Statistic Approach***   
A score is computed by counting sequences of lengths 5, 6, and 7 from the last SELEX cycle. 
These scores are averaged and normalized (0-4), then combined with the neural network and XGBoost predictions.

**Conclusion**✔️  
-
This study presents a novel approach for predicting RNA binding intensities from HTR-SELEX data using a hybrid model that integrates deep learning and statistical methods.   
Our neural network architecture effectively captures local sequence patterns through Conv1D layers, resulting in accurate predictions of binding scores.   
The incorporation of the **XGBoost algorithm further enhances performance** by leveraging probability vectors derived from the neural network.   
Additionally, **generating negative data through random nucleotide sampling and employing padding techniques ensured robust model training**.  

