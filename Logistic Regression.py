#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#five fold CV taken from previous projects
def get_five_folds(data_points):
    data_points = data_points.sample(frac=1).reset_index(drop=True)
    num_columns = len(data_points.columns)
    num_rows = len(data_points.index)
    fold0 = pd.DataFrame(columns=(data_points.columns))
    fold1 = pd.DataFrame(columns=(data_points.columns))
    fold2 = pd.DataFrame(columns=(data_points.columns))
    fold3 = pd.DataFrame(columns=(data_points.columns))
    fold4 = pd.DataFrame(columns=(data_points.columns))
 
    actual_class_column = num_columns - 1
    unique_class_list_df = data_points.iloc[:,actual_class_column]
    unique_class_list_df = unique_class_list_df.sort_values()
    unique_class_list_np = unique_class_list_df.unique()
    unique_class_list_df = unique_class_list_df.drop_duplicates()
    unique_class_list_np_size = unique_class_list_np.size
 
    for unique_class_list_np_idx in range(0, unique_class_list_np_size):
        counter = 0
        for row in range(0, num_rows):
            if unique_class_list_np[unique_class_list_np_idx] == (data_points.iloc[row,actual_class_column]):
                    if counter == 0:
                        new_row = data_points.iloc[row,:]
                        fold0.loc[len(fold0)] = new_row
                        counter += 1
                    elif counter == 1:
                        new_row = data_points.iloc[row,:]
                        fold1.loc[len(fold1)] = new_row
                        counter += 1
                    elif counter == 2:
                        new_row = data_points.iloc[row,:]
                        fold2.loc[len(fold2)] = new_row
                        counter += 1
                    elif counter == 3:
                        new_row = data_points.iloc[row,:]
                        fold3.loc[len(fold3)] = new_row
                        counter += 1
                    else:
                        new_row = data_points.iloc[row,:]
                        fold4.loc[len(fold4)] = new_row
                        counter = 0        
    return fold0, fold1, fold2, fold3, fold4


# In[51]:


def sigmoid(z):
    return 1.0/(1 + np.exp(-z))
 
def logistic_gradient_descent(train_set):
    num_columns_train_set = train_set.shape[1]
    num_rows_train_set = train_set.shape[0]
 
    #Number of features from the train data
    x = train_set[:,:(num_columns_train_set - 1)]
    num_features = x.shape[1]
    #Actual classes from the train data
    actual_class = train_set[:,(num_columns_train_set - 1)]
    #tunable learning rate
    LEARNING_RATE = 0.01
    #maximum number of iterations
    max_iter = 10000
    iter = 0
    #boolean to determine if we have passed the max iterations
    max_iter_reached = False
    #setting gradient tolerence level for a stopping point of for loop
    lowest_gradient = 0.001
    gradient_norm = None
    #boolean to determine if the minimum of the cost function
    converged = False
    #vector of weights
    weights = np.random.uniform(-0.01,0.01,(num_features))
    weight_changes = None
    #loop runs until not converged or max iterations reached
    while(not(converged) and not(max_iter_reached)):         
        #weight change vector
        weight_changes = np.zeros(num_features)
        for data_point in range(0, num_rows_train_set):
            #weighted sum of the features
            output = np.dot(weights, x[data_point,:])
            #probability that this datapoint belongs to the positive class
            y =  sigmoid(output)
            #difference in accuracy
            difference = (actual_class[data_point] - y)
            #product of difference and attribute vector
            product = np.multiply(x[data_point,:], difference)
            #update the weight changes
            weight_changes = np.add(weight_changes,product)   
        #step size y weights and learning rate
        step_size = np.multiply(weight_changes, LEARNING_RATE)
        #update weights
        weights = np.add(weights, step_size)
        gradient_norm = np.linalg.norm(weight_changes)
        #print statements for demo
        #print("New Weights")
        #print(weights)
        #print("New Gradient")
        #print(gradient_norm)
        
        if (gradient_norm < lowest_gradient):
            converged = True
        iter += 1
        if (iter > max_iter):
            max_iter_reached = True
    return weights
 

def logistic_regression(train_set, test_set):
    #remove datapoint id
    train_set = train_set.drop(train_set.columns[[0]], axis=1)
    test_set = test_set.drop(test_set.columns[[0]], axis=1)
    #unique classes
    list_of_unique_classes = pd.unique(train_set["Actual Class"])
    #changes all class values into numerical values
    for cl in range(0, len(list_of_unique_classes)):
        train_set["Actual Class"].replace(list_of_unique_classes[cl], cl ,inplace=True)
        test_set["Actual Class"].replace(list_of_unique_classes[cl], cl ,inplace=True)
    #column of 1s in column 0 of both the train and test sets. This is the bias and helps with gradient descent
    train_set.insert(0, "Bias", 1)
    test_set.insert(0, "Bias", 1)
    #convert dataframes to numpy arrays
    np_train_set = train_set.values
    np_test_set = test_set.values
    #test set adds predictions
    test_set = test_set.reindex(columns=[*test_set.columns.tolist(), 'Predicted Class', 'Prediction Correct?'])
 
    #train
    num_columns_train_set = np_train_set.shape[1]
    num_rows_train_set = np_train_set.shape[0] 
    #train set for each unique class
    trainsets = []
    for cl in range(0, len(list_of_unique_classes)):
        temp = np.copy(np_train_set)
        for row in range(0, num_rows_train_set):
            if (temp[row, (num_columns_train_set - 1)]) == cl:
                temp[row, (num_columns_train_set - 1)] = 1
            else:
                temp[row, (num_columns_train_set - 1)] = 0
        trainsets.append(temp)
    #calculate and store weights for each class
    weights_for_each_class = [] 
    for cl in range(0, len(list_of_unique_classes)):
        weights_for_this_class = logistic_gradient_descent(trainsets[cl])
        weights_for_each_class.append(weights_for_this_class)
 
    #TESTING
    num_columns_test_set = np_test_set.shape[1]
    num_rows_test_set = np_test_set.shape[0]
    #gets features and class from test set
    x = np_test_set[:,:(num_columns_test_set - 1)]
    num_features = x.shape[1]
    actual_class = np_test_set[:,(num_columns_test_set - 1)]
    for data_point in range(0,  num_rows_test_set): 
        #probabilities array
        p = []
        #calculate each probability
        for cl in range(0, len(list_of_unique_classes)):
            output = np.dot(weights_for_each_class[cl], x[data_point,:])
            #does not use sigmoid function like logistic regression
            this_probability = sigmoid(output)
            p.append(this_probability)
        #predicted class
        most_likely_class = p.index(max(p))
        test_set.loc[data_point, "Predicted Class"] = most_likely_class
        #checks if prediction is correct
        if test_set.loc[data_point, "Actual Class"] == test_set.loc[data_point, "Predicted Class"]:
            test_set.loc[data_point, "Prediction Correct?"] = 1
        else:
            test_set.loc[data_point, "Prediction Correct?"] = 0
    accuracy = (test_set["Prediction Correct?"].sum())/(len(test_set.index))
    predictions = test_set
 
    #prepares data in a returnable format
    for cl in range(0, len(list_of_unique_classes)):
        predictions["Actual Class"].replace(cl, list_of_unique_classes[cl] ,inplace=True)
        predictions["Predicted Class"].replace(cl, list_of_unique_classes[cl] ,inplace=True)
    predictions['Prediction Correct?'] = predictions['Prediction Correct?'].map({1: "Yes", 0: "No"})
    weights_for_each_class = pd.DataFrame(np.row_stack(weights_for_each_class))
    for cl in range(0, len(list_of_unique_classes)):
        row_name = str(list_of_unique_classes[cl] + " weights")        
        weights_for_each_class.rename(index={cl:row_name}, inplace=True)
    train_set_names = list(train_set.columns.values)
    train_set_names.pop()
    for col in range(0, len(train_set_names)):
        col_name = str(train_set_names[col])        
        weights_for_each_class.rename(columns={col:col_name}, inplace=True)
    num_datapoints_test = len(test_set.index)
    
    return accuracy, predictions, weights_for_each_class, num_datapoints_test


# In[54]:


def runLogisticRegression(path):
   pd_data_set = pd.read_csv(path, sep = ",")
   num_folds = 5
   fold0, fold1, fold2, fold3, fold4 = get_five_folds(pd_data_set)
   train_dataset = None
   test_dataset = None
   accuracy_statistics = np.zeros(num_folds)
   for fold in range(0, num_folds):
       print()
       print("Running fold " + str(fold + 1) + " ...")
       print()
       if fold == 0:
           test_dataset = fold0
           train_dataset = pd.concat([fold1, fold2, fold3, fold4], ignore_index=True, sort=False)                
       elif fold == 1:
           test_dataset = fold1
           train_dataset = pd.concat([fold0, fold2, fold3, fold4], ignore_index=True, sort=False) 
       elif fold == 2:
           test_dataset = fold2
           train_dataset = pd.concat([fold0, fold1, fold3, fold4], ignore_index=True, sort=False) 
       elif fold == 3:
           test_dataset = fold3
           train_dataset = pd.concat([fold0, fold1, fold2, fold4], ignore_index=True, sort=False) 
       else:
           test_dataset = fold4
           train_dataset = pd.concat([fold0, fold1, fold2, fold3], ignore_index=True, sort=False) 
        
       accuracy, predictions, weights_for_each_class, num_datapoints_test = (logistic_regression(train_dataset,test_dataset))

       print("Accuracy:")
       print(str(accuracy * 100) + "%")
       print()
       print("Classifications:")
       print(predictions)
       print()
       print("Learned Model:")
       print(weights_for_each_class)
       print()
       accuracy_statistics[fold] = accuracy

   accuracy = np.mean(accuracy_statistics)
   accuracy *= 100
   print("Accuracy Statistics for All folds:")
   print(accuracy_statistics) 
   print("Classification Accuracy : " + str(accuracy) + "%")


# In[52]:


#gradient descent for adaline
def adaline_gradient_descent(train_set):
    num_columns_train_set = train_set.shape[1]
    num_rows_train_set = train_set.shape[0]
 
    #Number of features from the train data
    x = train_set[:,:(num_columns_train_set - 1)]
    num_features = x.shape[1]
    #Actual classes from the train data
    actual_class = train_set[:,(num_columns_train_set - 1)]
    #tunable learning rate
    LEARNING_RATE = 0.01 
    #maximum number of iterations
    max_iter = 10000
    iter = 0
    #boolean to determine if we have passed the max iterations
    max_iter_reached = False
    #setting gradient tolerence level for a stopping point of for loop
    lowest_gradient = 0.001
    gradient_norm = None 
    #boolean to determine if the minimum of the cost function
    converged = False 
    #vector of weights
    weights = np.random.uniform(-0.01,0.01,(num_features))
    weight_changes = None 
    #loop runs until not converged or max iterations reached
    while(not(converged) and not(max_iter_reached)):         
        #weight change vector
        weight_changes = np.zeros(num_features)
        for data_point in range(0, num_rows_train_set):
            #weighted sum of the features
            output = np.dot(weights, x[data_point,:])
            #probability that this datapoint belongs to the positive class
            y =  output
            #difference in accuracy
            difference = (actual_class[data_point] - y)
            #product of difference and attribute vector and learning rate
            product = np.multiply(x[data_point,:], difference)*LEARNING_RATE
            #update the weight changes
            weight_changes = np.add(weight_changes,product)   
        #step size y weights and learning rate
        step_size = np.multiply(weight_changes, LEARNING_RATE)
        #update weights
        weights = np.add(weights, step_size)
        gradient_norm = np.linalg.norm(weight_changes)
        #print statements for demo
        #print("New Weights")
        #print(weights)
        #print("New Gradient")
        #print(gradient_norm)
        
        if (gradient_norm < lowest_gradient):
            converged = True
        iter += 1
        if (iter > max_iter):
            max_iter_reached = True
    return weights
 

def adaline(train_set, test_set):
    #remove datapoint id
    train_set = train_set.drop(train_set.columns[[0]], axis=1)
    test_set = test_set.drop(test_set.columns[[0]], axis=1)
    #unique classes
    list_of_unique_classes = pd.unique(train_set["Actual Class"]) 
    #changes all class values into numerical values
    for cl in range(0, len(list_of_unique_classes)):
        train_set["Actual Class"].replace(list_of_unique_classes[cl], cl ,inplace=True)
        test_set["Actual Class"].replace(list_of_unique_classes[cl], cl ,inplace=True) 
    #column of 1s in column 0 of both the train and test sets. This is the bias and helps with gradient descent
    train_set.insert(0, "Bias", 1)
    test_set.insert(0, "Bias", 1) 
    #convert dataframes to numpy arrays
    np_train_set = train_set.values
    np_test_set = test_set.values 
    #test set adds predictions
    test_set = test_set.reindex(columns=[*test_set.columns.tolist(), 'Predicted Class', 'Prediction Correct?'])
 
    #train
    num_columns_train_set = np_train_set.shape[1]
    num_rows_train_set = np_train_set.shape[0] 
    #train set for each unique class
    trainsets = []
    for cl in range(0, len(list_of_unique_classes)):
        temp = np.copy(np_train_set)
        for row in range(0, num_rows_train_set):
            if (temp[row, (num_columns_train_set - 1)]) == cl:
                temp[row, (num_columns_train_set - 1)] = 1
            else:
                temp[row, (num_columns_train_set - 1)] = 0
        trainsets.append(temp) 
    #calculate and store weights for each class
    weights_for_each_class = [] 
    for cl in range(0, len(list_of_unique_classes)):
        weights_for_this_class = adaline_gradient_descent(trainsets[cl])
        weights_for_each_class.append(weights_for_this_class)
 
    #TESTING
    num_columns_test_set = np_test_set.shape[1]
    num_rows_test_set = np_test_set.shape[0] 
    #gets features and class from test set
    x = np_test_set[:,:(num_columns_test_set - 1)]
    num_features = x.shape[1]
    actual_class = np_test_set[:,(num_columns_test_set - 1)] 
    for data_point in range(0,  num_rows_test_set):
        #probabilities array
        p = []
        #calculate each probability
        for cl in range(0, len(list_of_unique_classes)):
            output = np.dot(weights_for_each_class[cl], x[data_point,:])
            #does not use sigmoid function like logistic regression
            this_probability = output
            p.append(this_probability)
        #predicted class
        most_likely_class = p.index(max(p))
        test_set.loc[data_point, "Predicted Class"] = most_likely_class
        #checks if prediction is correct
        if test_set.loc[data_point, "Actual Class"] == test_set.loc[data_point, "Predicted Class"]:
            test_set.loc[data_point, "Prediction Correct?"] = 1
        else:
            test_set.loc[data_point, "Prediction Correct?"] = 0
    accuracy = (test_set["Prediction Correct?"].sum())/(len(test_set.index))
    predictions = test_set
 
    #prepares data in a returnable format
    for cl in range(0, len(list_of_unique_classes)):
        predictions["Actual Class"].replace(cl, list_of_unique_classes[cl] ,inplace=True)
        predictions["Predicted Class"].replace(cl, list_of_unique_classes[cl] ,inplace=True)
    predictions['Prediction Correct?'] = predictions['Prediction Correct?'].map({1: "Yes", 0: "No"})
    weights_for_each_class = pd.DataFrame(np.row_stack(weights_for_each_class))
    for cl in range(0, len(list_of_unique_classes)):
        row_name = str(list_of_unique_classes[cl] + " weights")        
        weights_for_each_class.rename(index={cl:row_name}, inplace=True)
    train_set_names = list(train_set.columns.values)
    train_set_names.pop()
    for col in range(0, len(train_set_names)):
        col_name = str(train_set_names[col])        
        weights_for_each_class.rename(columns={col:col_name}, inplace=True)
    num_datapoints_test = len(test_set.index)
    
    return accuracy, predictions, weights_for_each_class, num_datapoints_test


# In[55]:


#same code as runLogistic Regression but for adaline
def runAdaline(path):
    pd_data_set = pd.read_csv(path, sep = ",")
    num_folds = 5
    fold0, fold1, fold2, fold3, fold4 = get_five_folds(pd_data_set)
    train_dataset = None
    test_dataset = None
    accuracy_statistics = np.zeros(num_folds)
    for fold in range(0, num_folds):
        print()
        print("Running fold " + str(fold + 1) + " ...")
        print()
        if fold == 0:
            test_dataset = fold0
            train_dataset = pd.concat([fold1, fold2, fold3, fold4], ignore_index=True, sort=False)                
        elif fold == 1:
            test_dataset = fold1
            train_dataset = pd.concat([fold0, fold2, fold3, fold4], ignore_index=True, sort=False) 
        elif fold == 2:
            test_dataset = fold2
            train_dataset = pd.concat([fold0, fold1, fold3, fold4], ignore_index=True, sort=False) 
        elif fold == 3:
            test_dataset = fold3
            train_dataset = pd.concat([fold0, fold1, fold2, fold4], ignore_index=True, sort=False) 
        else:
            test_dataset = fold4
            train_dataset = pd.concat([fold0, fold1, fold2, fold3], ignore_index=True, sort=False) 
         
        accuracy, predictions, weights_for_each_class, num_datapoints_test = (adaline(train_dataset,test_dataset))
 
        print("Accuracy:")
        print(str(accuracy * 100) + "%")
        print()
        print("Classifications:")
        print(predictions)
        print()
        print("Learned Model:")
        print(weights_for_each_class)
        print()
        accuracy_statistics[fold] = accuracy
 
    accuracy = np.mean(accuracy_statistics)
    accuracy *= 100
    print("Accuracy Statistics for All folds:")
    print(accuracy_statistics) 
    print("Classification Accuracy : " + str(accuracy) + "%")


# In[43]:


#datasets after preprocessing
iris = "/Users/reza/Desktop/Stuff3/JHU/MachineLearning/Project 4/iris.txt"
cancer = "/Users/reza/Desktop/Stuff3/JHU/MachineLearning/Project 4/breast_cancer.txt"
glass = "/Users/reza/Desktop/Stuff3/JHU/MachineLearning/Project 4/glass.txt"
soybean = "/Users/reza/Desktop/Stuff3/JHU/MachineLearning/Project 4/soybean.txt"
vote = "/Users/reza/Desktop/Stuff3/JHU/MachineLearning/Project 4/vote.txt"


# In[47]:


#for demo
#runLogisticRegression(iris)


# In[50]:


#for demo
#runAdaline(cancer)


# In[ ]:


##running algorithms on every dataset
runLogisticRegression(cancer)


# In[ ]:


runAdaline(cancer)


# In[ ]:


runLogisticRegression(glass)


# In[ ]:


runAdaline(glass)


# In[67]:


runLogisticRegression(iris)


# In[68]:


runAdaline(iris)


# In[ ]:


runLogisticRegression(soybean)


# In[ ]:


runAdaline(soybean)


# In[ ]:


runLogisticRegression(vote)


# In[ ]:


runAdaline(vote)

