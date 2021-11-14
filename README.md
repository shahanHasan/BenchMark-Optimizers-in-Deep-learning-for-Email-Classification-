# A-Comparative-Analysis-of-Optimizers in Bi-Directional Recurrent Neural Networks for NLP tasks

## In this work we train several deep learning algorithms using exactly same parameters other than the optimizer. We benchmark the optimizers based on the performance metrics of the models
## specifically training and validation loss. We train the models on several popular benchmark spam email datasets mentioned below. 

### Dataset :
1. Enron
2. Ling Spam
3. Spam Assasin
4. Spam Base

### Feature Extraction :
    - Word Embedding (50 dimentional)

### Otimizers :
1. Adam
2. Nadam
3. SGD
4. RMSprop
5. Adagrad
6. Adadelta 
7. Ftrl 
8. SGD momentum 
9. SGD Nesterov Momentum
10. Adam Weight Decay 

### Optimizers to be tried later :
1. AMSgrad
2. AMSbound
3. Adabound
4. Radam
5. Adamax
6. Adabelief

### Deep Learning Algorithms : 
1. Bi-Directional Recurrent Neural Network
2. Bi-Directional Long Short Term Memory
3. Bi-Directional Gated Recurrent Unit

### Directory Structure of the repo :

```
.
|-- Comparison
|   |-- Comparison_BarCharts
|   |   |-- LogNorm_Loss_barplot.jpeg
|   |   |-- LogNorm_Test_Acc_barplot.jpeg
|   |   |-- LogNorm_Train_Acc_barplot.jpeg
|   |   |-- Test_Acc_barplot.jpeg
|   |   |-- Test_Loss_barplot.jpeg
|   |   `-- Train_Acc_barplot.jpeg
|   |-- Previous
|   |   |-- Adam_comparison.csv
|   |   |-- All_optimizers.csv
|   |   |-- Nadam_comparison.csv
|   |   `-- RMS_comparison.csv
|   |-- Average_Overall_dataset.csv
|   |-- Enron_Bi-GRU_comparison.csv
|   |-- Enron_Bi-LSTM_comparison.csv
|   |-- Enron_BRNN_comparison.csv
|   |-- Enron_comparison.csv
|   |-- Enron_optim.csv
|   |-- LingSpam_Bi-GRU_comparison.csv
|   |-- LingSpam_Bi-LSTM_comparison.csv
|   |-- LingSpam_BRNN_comparison.csv
|   |-- LingSpam_comparison.csv
|   |-- LingSpam_optim.csv
|   |-- SpamAssasin_Bi-GRU_comparison
|   |-- SpamAssasin_Bi-GRU_comparison.csv
|   |-- SpamAssasin_Bi-LSTM_comparison
|   |-- SpamAssasin_Bi-LSTM_comparison.csv
|   |-- SpamAssasin_BRNN_comparison.csv
|   |-- SpamAssasin_comparison.csv
|   `-- SpamAssasin_optim.csv
|-- Datasets
|   |-- Enron
|   |   |-- PROCESSED.csv
|   |   `-- spam(madeof).csv
|   |-- lingspam
|   |   |-- messages.csv
|   |   `-- PROCESSED.csv
|   |-- SpamAssasin
|   |   |-- Processed
|   |   |   `-- PROCESSED.csv
|   |   |-- 20021010_easy_ham.tar.bz2
|   |   |-- 20021010_hard_ham.tar.bz2
|   |   |-- 20021010_spam.tar.bz2
|   |   |-- 20030228_easy_ham_2.tar.bz2
|   |   |-- 20030228_easy_ham.tar.bz2
|   |   |-- 20030228_hard_ham.tar.bz2
|   |   |-- 20030228_spam_2.tar.bz2
|   |   |-- 20030228_spam.tar.bz2
|   |   |-- 20050311_spam_2.tar.bz2
|   |   |-- obsolete.zip
|   |   `-- readme.html
|   `-- spambase
|       |-- PROCESSED.csv
|       |-- spambase.data
|       |-- spambase.DOCUMENTATION
|       `-- spambase.names
|-- Heatmaps
|   |-- ENRON
|   |-- LINGSPAM
|   |-- Previous
|   |   |-- Bi-LSTM-heatmap.jpeg
|   |   |-- Bi-LSTM-nadam-heatmap.jpeg
|   |   |-- Bi-LSTM-rms-heatmap.jpeg
|   |   |-- gru-heatmap.jpeg
|   |   |-- gru-nadam-heatmap.jpeg
|   |   |-- gru-rms-heatmap.jpeg
|   |   |-- LSTM-heatmap.jpeg
|   |   |-- LSTM-nadam-heatmap.jpeg
|   |   `-- LSTM-rms-heatmap.jpeg
|   |-- SPAMASSASIN
|   `-- SPAMBASE
|-- Models
|   |-- ENRON
|   |   |-- Enron_history.pkl
|   |   |-- Enron.pkl
|   |   `-- Enron_y_pred.pkl
|   |-- LINGSPAM
|   |   |-- LingSpam_history.pkl
|   |   |-- LingSpam.pkl
|   |   `-- LingSpam_y_pred.pkl
|   |-- Previous
|   |   |-- Bi_LSTM.h5
|   |   |-- Bi_LSTM-nadam.h5
|   |   |-- Bi_LSTM-rms.h5
|   |   |-- GRU.h5
|   |   |-- GRU_nadam.h5
|   |   |-- GRU_rms.h5
|   |   |-- LSTM.h5
|   |   |-- LSTM_nadam.h5
|   |   `-- LSTM_rms.h5
|   |-- SPAMASSASIN
|   |   |-- SpamAssasin_history.pkl
|   |   |-- SpamAssasin.pkl
|   |   `-- SpamAssasin_y_pred.pkl
|   `-- SPAMBASE
|-- Previous
|   |-- Adam_LSTM,BiLSTM,GRU.ipynb
|   |-- Nadam_LSTM,BiLSTM,GRU.ipynb
|   `-- Rmsprop_LSTM,BiLSTM,GRU.ipynb
|-- Visuals
|   |-- ENRON
|   |   |-- Accuracy_Loss
|   |   |   |-- Enron_accuracy_Bi-GRU.jpeg
|   |   |   |-- Enron_accuracy_Bi-LSTM.jpeg
|   |   |   |-- Enron_accuracy_BRNN.jpeg
|   |   |   |-- Enron_accuracy_BRNN_legend.jpeg
|   |   |   |-- Enron_loss_Bi-GRU.jpeg
|   |   |   |-- Enron_loss_Bi-LSTM.jpeg
|   |   |   |-- Enron_loss_BRNN.jpeg
|   |   |   |-- Enron_loss_BRNN_legend.jpeg
|   |   |   |-- Enron_val_accuracy_Bi-GRU.jpeg
|   |   |   |-- Enron_val_accuracy_Bi-LSTM.jpeg
|   |   |   |-- Enron_val_accuracy_BRNN.jpeg
|   |   |   |-- Enron_val_accuracy_BRNN_legend.jpeg
|   |   |   |-- Enron_val_loss_Bi-GRU.jpeg
|   |   |   |-- Enron_val_loss_Bi-LSTM.jpeg
|   |   |   |-- Enron_val_loss_BRNN.jpeg
|   |   |   `-- Enron_val_loss_BRNN_legend.jpeg
|   |   |-- AU-ROC
|   |   |-- PieChart.jpeg
|   |   |-- wordcloud_ham.jpeg
|   |   |-- word_cloud_overall.jpeg
|   |   `-- wordcloud_spam.jpeg
|   |-- LINGSPAM
|   |   |-- Accuracy_Loss
|   |   |   |-- LingSpam_accuracy_Bi-GRU.jpeg
|   |   |   |-- LingSpam_accuracy_Bi-LSTM.jpeg
|   |   |   |-- LingSpam_accuracy_BRNN.jpeg
|   |   |   |-- LingSpam_accuracy_BRNN_legend.jpeg
|   |   |   |-- LingSpam_loss_Bi-GRU.jpeg
|   |   |   |-- LingSpam_loss_Bi-LSTM.jpeg
|   |   |   |-- LingSpam_loss_BRNN.jpeg
|   |   |   |-- LingSpam_loss_BRNN_legend.jpeg
|   |   |   |-- LingSpam_val_accuracy_Bi-GRU.jpeg
|   |   |   |-- LingSpam_val_accuracy_Bi-LSTM.jpeg
|   |   |   |-- LingSpam_val_accuracy_BRNN.jpeg
|   |   |   |-- LingSpam_val_accuracy_BRNN_legend.jpeg
|   |   |   |-- LingSpam_val_loss_Bi-GRU.jpeg
|   |   |   |-- LingSpam_val_loss_Bi-LSTM.jpeg
|   |   |   |-- LingSpam_val_loss_BRNN.jpeg
|   |   |   `-- LingSpam_val_loss_BRNN_legend.jpeg
|   |   |-- AU-ROC
|   |   |-- PieChart.jpeg
|   |   |-- wordcloud_ham.jpeg
|   |   |-- word_cloud_overall.jpeg
|   |   `-- wordcloud_spam.jpeg
|   |-- Previous
|   |   |-- Bi-LSTM-acc-loss.jpeg
|   |   |-- Bi-LSTM_AUC.jpeg
|   |   |-- Bi-LSTM-nadam-acc-loss.jpeg
|   |   |-- Bi-LSTM-nadam_AUC.jpeg
|   |   |-- Bi-LSTM-rms-acc-loss.jpeg
|   |   |-- Bi-LSTM-rms_AUC.jpeg
|   |   |-- gru-acc-loss.jpeg
|   |   |-- gru_AUC.jpeg
|   |   |-- gru-nadam-acc-loss.jpeg
|   |   |-- gru_nadam_AUC.jpeg
|   |   |-- gru-rms-acc-loss.jpeg
|   |   |-- gru_rms_AUC.jpeg
|   |   |-- LSTM-acc-loss.jpeg
|   |   |-- LSTM_AUC.jpeg
|   |   |-- LSTM-nadam-acc-loss.jpeg
|   |   |-- LSTM_nadam_AUC.jpeg
|   |   |-- LSTM-rms-acc-loss.jpeg
|   |   `-- LSTM_rms_AUC.jpeg
|   |-- SPAMASSASIN
|   |   |-- Accuracy_Loss
|   |   |   |-- SpamAssasin_accuracy_Bi-GRU.jpeg
|   |   |   |-- SpamAssasin_accuracy_Bi-LSTM.jpeg
|   |   |   |-- SpamAssasin_accuracy_BRNN.jpeg
|   |   |   |-- SpamAssasin_accuracy_BRNN_legend.jpeg
|   |   |   |-- SpamAssasin_loss_Bi-GRU.jpeg
|   |   |   |-- SpamAssasin_loss_Bi-LSTM.jpeg
|   |   |   |-- SpamAssasin_loss_BRNN.jpeg
|   |   |   |-- SpamAssasin_loss_BRNN_legend.jpeg
|   |   |   |-- SpamAssasin_val_accuracy_Bi-GRU.jpeg
|   |   |   |-- SpamAssasin_val_accuracy_Bi-LSTM.jpeg
|   |   |   |-- SpamAssasin_val_accuracy_BRNN.jpeg
|   |   |   |-- SpamAssasin_val_accuracy_BRNN_legend.jpeg
|   |   |   |-- SpamAssasin_val_loss_Bi-GRU.jpeg
|   |   |   |-- SpamAssasin_val_loss_Bi-LSTM.jpeg
|   |   |   |-- SpamAssasin_val_loss_BRNN.jpeg
|   |   |   `-- SpamAssasin_val_loss_BRNN_legend.jpeg
|   |   |-- AU-ROC
|   |   |-- PieChart.jpeg
|   |   |-- wordcloud_ham.jpeg
|   |   |-- word_cloud_overall.jpeg
|   |   `-- wordcloud_spam.jpeg
|   `-- SPAMBASE
|       |-- Accuracy_Loss
|       `-- AU-ROC
|-- Comparison.ipynb
|-- ENRON.ipynb
|-- ENRON, LINGSPAM, SPAMASSASIN DATA PREPARATION.ipynb
|-- LICENSE
|-- LingSpam.ipynb
|-- README.md
|-- SpamAssasin.ipynb
|-- SpamBase.ipynb
`-- utils.py

```


### TO DO :
1. Parameter Optimization
    - Hyper parameter tuning (learning rate, momentum, decay, betas etc)
    - Model Architecture Tuning
    - Hidden Units Tuning
    
2. Try Other Deel learning algorithms : CNN, ANN, CNN-BI-LSTM, BERT, Transformers     
3. Try other datasets , problems (Classification, Generative, Reinforcement) etc 