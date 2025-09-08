# """
# This class will provide utilities to evaluate the performance of the classifier. After training a model, we need to see how well it performs on unseen data.
# The metrics I will cover will be Accuracy, Precision, Recall, F1 score and confusion matrix.
# Methods in this class will be static so that there is no need to create an object.

# Accuracy: proportion of articles correctly classified
# Precision: how many articles predicted fake were actually fake
# Recall: Of all the fake news articles, how many did the model actually catch.
# F1 Score: mean of precision and recall.
# Confusion Matrix: 2x2 table showing counts of [TRUE REAL, FALSE FAKE] vs [FALSE REAL, TRUE FAKE] predictions . Rows: Actual class (Real,Fake) - Columns: Predicted class(Real,Fake). It shows how many real news were correctly and incorrectly classified, and same for fake news
# """
# from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix
# import seaborn as sns
# import matplotlib.pyplot as plt


# class Evaluator:
    

#     # This method will evaluate metrics discussed in the documentation above
#     @staticmethod
#     def evaluate(y_true, y_pred):
#         if len(y_true) != len(y_pred):
#             raise ValueError("Mismatch in number of true and predicted labels")
        
#         metrics = {}
#         metrics['accuracy'] = accuracy_score(y_true, y_pred)
#         metrics['precision'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0) #pos_label is 0 as we are interested in fake news which is 0
#         metrics['recall'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)# zero_Division handle cases where the formula divides by 0
#         metrics['f1_score'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

#         return metrics
    
    




#     #This method will give a confusion matrix values as a 2D array in the format: [[TN, FP], [FN, TP]]
#     @staticmethod
#     def confusion_matrix_values(y_true, y_pred):
#         if len(y_true) != len(y_pred):
#             raise ValueError("Mismatch in number of true and predicted labels")
        
#         cm = confusion_matrix(y_true, y_pred, labels = [0,1])
#         return cm

#     #This method will use seaborn heatmap to visually show the confusion matrix

#     @staticmethod
#     def plot_confusion_matrix(y_true, y_pred):
#         cm = Evaluator.confusion_matrix_values(y_true, y_pred)

#         plt.figure(figsize= (5, 4))
#         sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels=['Pred: Fake', 'Pred: Real'], yticklabels=['Actual: Fake', 'Actual: Real'])

#         plt.title("Confusion Matrix")
#         plt.xlabel("Predicted Class")
#         plt.ylabel("Actual Class")
#         plt.tight_layout()
#         plt.show()








"""
This class will provide utilities to evaluate the performance of the classifier. After training a model, we need to see how well it performs on unseen data.
The metrics I will cover will be Accuracy, Precision, Recall, F1 score and confusion matrix.
Methods in this class will be static so that there is no need to create an object.

Accuracy: proportion of articles correctly classified
Precision: how many articles predicted fake were actually fake
Recall: Of all the fake news articles, how many did the model actually catch.
F1 Score: mean of precision and recall.
Confusion Matrix: 2x2 table showing counts of [TRUE REAL, FALSE FAKE] vs [FALSE REAL, TRUE FAKE] predictions . Rows: Actual class (Real,Fake) - Columns: Predicted class(Real,Fake). It shows how many real news were correctly and incorrectly classified, and same for fake news
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class Evaluator:
    

    # This method will evaluate metrics discussed in the documentation above
    @staticmethod
    def evaluate(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("Mismatch in number of true and predicted labels")
        
        # Convert string labels to numeric if needed
        if isinstance(y_true[0], str):
            label_map = {'FAKE': 0, 'REAL': 1}
            y_true = [label_map[label] for label in y_true]
            y_pred = [label_map[label] for label in y_pred]
        
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0) #pos_label is 0 as we are interested in fake news which is 0
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)# zero_Division handle cases where the formula divides by 0
        metrics['f1_score'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

        return metrics
    
    




    #This method will give a confusion matrix values as a 2D array in the format: [[TN, FP], [FN, TP]]
    @staticmethod
    def confusion_matrix_values(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("Mismatch in number of true and predicted labels")
        
        # Convert string labels to numeric if needed
        if isinstance(y_true[0], str):
            label_map = {'FAKE': 0, 'REAL': 1}
            y_true = [label_map[label] for label in y_true]
            y_pred = [label_map[label] for label in y_pred]
        
        cm = confusion_matrix(y_true, y_pred, labels = [0,1])
        return cm

    #This method will use seaborn heatmap to visually show the confusion matrix

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred):
        # Convert string labels to numeric if needed
        if isinstance(y_true[0], str):
            label_map = {'FAKE': 0, 'REAL': 1}
            y_true = [label_map[label] for label in y_true]
            y_pred = [label_map[label] for label in y_pred]
        
        cm = Evaluator.confusion_matrix_values(y_true, y_pred)

        plt.figure(figsize= (5, 4))
        sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', xticklabels=['Pred: Fake', 'Pred: Real'], yticklabels=['Actual: Fake', 'Actual: Real'])

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.tight_layout()
        plt.show()