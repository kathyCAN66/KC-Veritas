**3.1.26**  
- read decision tree tutorial
- started coding toy project
- preliminary results

**Training on Random dataset:**
- Gini Index:
  - Results (head): ['color' 'color' 'color' 'pos' 'color' 'color' 'either' 'either' 'color' 'pos']
  - Accuracy: 29.666666666666668
  - Confusion Matrix:

[[29 31 38]  
[35 26 34]  
[49 34 34]]  

  - Classification Report:

                      precision    recall  f1-score   support
               color       0.26      0.33      0.29        88
              either       0.29      0.27      0.28        95
                 pos       0.35      0.29      0.32       117   
            accuracy                           0.30       300
           macro avg       0.30      0.30      0.30       300 
        weighted avg       0.30      0.30      0.30       300
                                                                                 
- Entropy:
  - Results: ['color' 'color' 'color' 'pos' 'color' 'color' 'either' 'either' 'color' 'either']
  - Accuracy: 30.333333333333336
  - Confusion Matrix:
    
[[28 31 29]  
[34 29 32]   
[47 36 34]]                                                                                                                                                   
  - Classification Report:

                      precision    recall  f1-score   support
               color       0.26      0.32      0.28        88
              either       0.30      0.31      0.30        95
                 pos       0.36      0.29      0.32       117
            accuracy                           0.30       300
           macro avg       0.31      0.30      0.30       300
        weighted avg       0.31      0.30      0.30       300
    
**Training on Entropy dataset:**         
- Gini Index:                                                                                                                                            - Results: ['either' 'either' 'either' 'pos' 'either' 'either' 'either' 'either' 'either' 'pos']
  - Accuracy: 82.0
  - Confusion Matrix:
      
[[0 34 0]  
[0 217 17]  
[0 3 29]]                                                                                                                                              

  - Classification Report:

                       precision    recall  f1-score   support
                color       0.00      0.00      0.00        34
               either       0.85      0.93      0.89       234
                  pos       0.63      0.91      0.74        32
             accuracy                           0.82       300
            macro avg       0.49      0.61      0.54       300
         weighted avg       0.73      0.82      0.77       300
      
- Entropy:                                                                                                                                               - Results: ['either' 'either' 'either' 'pos' 'either' 'either' 'either' 'either' 'either' 'pos']
  - Accuracy: 81.66666666666667
  - Confusion Matrix:

[[0 34 0]  
[0 218 16]    
[0 5 27]]

  - Classification Report:
                                                                                                                       
                      precision    recall  f1-score   support  
               color       0.00      0.00      0.00        34    
              either       0.85      0.93      0.89       234
                 pos       0.63      0.84      0.72        32
            accuracy                           0.82       300
           macro avg       0.49      0.59      0.54       300
        weighted avg       0.73      0.82      0.77       300

**Training on Split Score dataset:**
- Gini Index:
  - Results: ['either' 'either' 'either' 'pos' 'either' 'either' 'either' 'color' 'color' 'pos']
  - Accuracy: 65.33333333333333
  - Confusion Matrix:

[[30 45 2]  
[10 115 27]        
[2 18 51]]  

  - Classification Report:

                      precision    recall  f1-score   support
               color       0.71      0.39      0.50        77  
              either       0.65      0.76      0.70       152
                 pos       0.64      0.72      0.68        71
            accuracy                           0.65       300
           macro avg       0.67      0.62      0.63       300
        weighted avg       0.66      0.65      0.64       300
         
- Entropy:
  - Results:['either' 'either' 'either' 'pos' 'color' 'either' 'either' 'color' 'color' 'pos']
  - Accuracy: 56.333333333333336
  - Confusion Matrix:
            
[[52 23 2]  
[59 66 27]  
[ 2 18 51]]                                                                                                                                                   
  - Classification Report:

                      precision    recall  f1-score   support 
               color       0.46      0.68      0.55        77
              either       0.62      0.43      0.51       152
                 pos       0.64      0.72      0.68        71
            accuracy                           0.56       300
           macro avg       0.57      0.61      0.58       300
        weighted avg       0.58      0.56      0.56       300
          
**Training on Weighted dataset:**         
- Gini Index:
  - Results: ['either' 'either' 'either' 'pos' 'either' 'either' 'either' 'color' 'color' 'pos']
  - Accuracy: 65.33333333333333
  - Confusion Matrix:
  
[[30 45 2]  
[10 115 27]  
[2 18 51]]

  - Classification Report:

                      precision    recall  f1-score   support
               color       0.71      0.39      0.50        77  
              either       0.65      0.76      0.70       152
                 pos       0.64      0.72      0.68        71
            accuracy                           0.65       300
           macro avg       0.67      0.62      0.63       300
        weighted avg       0.66      0.65      0.64       300

- Entropy:
  - Results: ['either' 'either' 'either' 'pos' 'color' 'either' 'either' 'color' 'color' 'pos']
  - Accuracy: 56.333333333333336
  - Confusion Matrix:

[[52 23 2]  
[59 66 27]  
[2 18 51]]

  - Classification Report:
 
                      precision    recall  f1-score   support
               color       0.46      0.68      0.55        77
              either       0.62      0.43      0.51       152
                 pos       0.64      0.72      0.68        71
            accuracy                           0.56       300
           macro avg       0.57      0.61      0.58       300
        weighted avg       0.58      0.56      0.56       300

**Training on Entropy Random dataset:**
- Gini Index:
  - Results: ['pos' 'color' 'pos' 'pos' 'color' 'color' 'color' 'color' 'pos' 'pos']
  - Accuracy: 60.66666666666667
  - Confusion Matrix:

[[117 35]  
[83 65]]

  - Classification Report:
    
                      precision    recall  f1-score   support
               color       0.58      0.77      0.66       152
                 pos       0.65      0.44      0.52       148
            accuracy                           0.61       300
           macro avg       0.62      0.60      0.59       300
        weighted avg       0.62      0.61      0.60       300
  
- Entropy:
  - Results: ['pos' 'color' 'pos' 'pos' 'color' 'color' 'color' 'color' 'pos' 'pos']
  - Accuracy: 61.33333333333333
  - Confusion Matrix:
     
[[117 35]        
[81 67]]  

  - Classification Report:
                                                                                                                            
                      precision    recall  f1-score   support
               color       0.59      0.77      0.67       152  
                 pos       0.66      0.45      0.54       148
            accuracy                           0.61       300
           macro avg       0.62      0.61      0.60       300
        weighted avg       0.62      0.61      0.60       300
          
**Training on Split Score Random dataset:**
- Gini Index:  
  - Results: ['color' 'color' 'color' 'pos' 'color' 'color' 'pos' 'color' 'color' 'pos']
  - Accuracy: 72.33333333333334
  - Confusion Matrix:                                                                                                                                              
[[137 40]  
[43 80]]

  - Classification Report:
                                                                                                                           
                      precision    recall  f1-score   support  
               color       0.76      0.77      0.77       177
                 pos       0.67      0.65      0.66       123    
            accuracy                           0.72       300
           macro avg       0.71      0.71      0.71       300
        weighted avg       0.72      0.72      0.72       300
          
- Entropy:
  - Results: ['color' 'color' 'color' 'pos' 'color' 'color' 'pos' 'color' 'color' 'pos']
  - Accuracy: 71.33333333333334
  - Confusion Matrix:  

[[137 40]  
[46 77]]

  - Classification Report:

                      precision    recall  f1-score   support 
               color       0.75      0.77      0.76       177
                 pos       0.66      0.63      0.64       123
            accuracy                           0.71       300
           macro avg       0.70      0.70      0.70       300
        weighted avg       0.71      0.71      0.71       300  

**Training on Weighted Random dataset:**       
- Gini Index:      
  - Results: ['color' 'color' 'color' 'pos' 'color' 'color' 'pos' 'color' 'color' 'pos']       
  - Accuracy: 63.33333333333333     
  - Confusion Matrix:

[[113  26]  
[ 84  77]]
 
  - Classification Report:

                      precision    recall  f1-score   support      
               color       0.57      0.81      0.67       139
                 pos       0.75      0.48      0.58       161
            accuracy                           0.63       300
           macro avg       0.66      0.65      0.63       300
        weighted avg       0.67      0.63      0.62       300

- Entropy:      
  - Results:['color' 'color' 'color' 'pos' 'color' 'color' 'pos' 'color' 'color' 'pos'] 
  - Accuracy: 63.33333333333333         
  - Confusion Matrix:  

[[113  26]     
[ 84  77]]
 
  - Classification Report:

                      precision    recall  f1-score   support
               color       0.57      0.81      0.67       139 
                 pos       0.75      0.48      0.58       161
            accuracy                           0.63       300
           macro avg       0.66      0.65      0.63       300
        weighted avg       0.67      0.63      0.62       300
    
