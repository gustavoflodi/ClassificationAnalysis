# Classification Analysis
Learning concepts and algorithms to solve Classification problems.

## Important Terms

It's important to get familiar with the names that repeat themselves often in this niche!

- Instances: represents an entity, an individual, a row, ...
    - Synonyms: data objects, tuples, rows, data points, examples, records, samples.
- Attributes: what is used to describe the instances and often to give them context.
    - Examples: age, profession, gender, education.
- Attribute Value: concrete ocurrence of an attribute, basically the values of the attributes.
- Class Label: the attribute which is missing and which we will predict.
    - Also known as Class Attribute.
- Class: the value itself of the class attribute, also missing.

## What is classification about?

> Classification is all about predicting a certain attribute from a data point, based on the history of the relation or dataset (the patterns described previously). The ultimate goal, therefore, is to predict the class of an unseen instance.

The attribute value is unknown prior to the beginning of the classification analysis, **but its classes are known**, i.e., the values one is trying to predict for the class label are pre-defined.

**It's a supervised learning problem!** But why? Simply because you have training data, so you have feedback involved for the algorithm and, when the model is built, you can apply it to unseen, test data.
    
    Ex.: Classification.

**In contrast, there are Machine Learning problems considered Unsupervised Learning**, which do not come along with training data, and the model construction, inference and use is based solely on test data.
    
    Ex.: Association, Clustering.
    
## 1R (One Rule) Algorithm

Let's get to know now, the (possibly) easiest method to predict classes.

> The whole idea is to predict a class attribute from just one known attribute, which has the least total error. Easy, right?

    1. Simply analyze each attribute at a time. When this attribute has a specific value, what is the common output in the class? 
    2. There! You have your rule, now count the errors made from this rule and sum it to the other specific values from this attribute. 
    3. Finally, compare it to the other attributes to define the one with the smallest number of errors.
    
## Model evaluation: how good is our first model?

We have the single rule, but how good is its prediction?

The objective at this stage is to define a parameter to judge our model and, later on, give the parameter a value, with the interest of, in the end, choosing an **optimized model**.

- **Evaluation does not happen with the training data**: this has too much variance, which means our model will be perfect for training data (actually too perfect), making it unsuitable for new, unseen data.
    - Also refered to as **overfitting**.
    
### Confusion Matrix

Matrix used to calculate important parameters and its values for model evaluation.

|                        | Positive Cases   | Negative Cases            |                                |
| -----------            | -----------      | -----------               | -----------                    |
|**Predicted Positives** | True Positives (TP)   | False Positives (FP) | **Precision**                  |
|**Predicted Negatives** | False Negatives (FN)  | True Negatives (TN)  | **Negative Prediction Value**  |
|                        | **Recall**            | **Specificity**      | **Accuracy**                   |
- Accuracy: parameter to calculate if the True predictions are high and the False ones are low.
    $$ (TP + TN) / (FP + FN) $$
    - Often not sufficient.
- Error Rate
    $$ 1 - Accuracy $$
- Precision: in all the predicted target class, how well was it predicted?
    $$ TP / (TP + FP) $$
- Negative Prediction Value: Precision for the negative predicted class.
    $$ TN / (TN + FN) $$
- Recall or Sensitivity: In all the actual target class, how well was it classified?
    $$ TP / (TP + FN) $$
    - Precision and Recall have mutual inference.
        - If the model is configured to hit more TPs, it will probably give more "false alarms" with FPs. Reducing Precision, but, at the same time, increasing Recall as the FNs did not change. This is called the **Precision-Recall Trade-off**.
        - Possible solution: $$ F-measure = 2*((Precision * Recall)/(Precision + Recall))  = (2TP) / (2TP + FN + FP) $$
- Specificity or Correct Rejection rate: Sensitivity for the negative class.
    $$ TN / (TN + FP) $$

    > Take a look again at the confusion matrix, the parameters are calculated from the values of the rows and columns!
    
## Naive Bayes Classification

A method for classification and **predictor selection**.

- The principle here is to assign a posterior probability to each predictor given the target class.

### Important terms

- A-priori probability: probability that event A happens.
- A-posteriori probability: probability that event A happens, given that B already occured.

### Bayes' Theorem

$$ P(A|B) = (P(B|A) * P(A)) / P(B) $$

*Probability that A occurs, given that B already occured, is equal to the probability of B given A, times probability of A, divided by probability of B.*

### Back to the classification: Steps

1. We have to classify an unknown test instance T.
2. We calculate the probability of observing T, given the hypothesis h that T belongs to the class ci.
3. h would be that the class is, for example, "TRUE" for wheter today will rain or not. And T is the instance with all its known attributes.
4. We get then that:
        $$ P(h|T) = (P(T|h) * P(h)) / P(T) $$
5. T being the vector of attributes known, we have to determine the probability for each of the attributes ak: P(ak|ci).
6. We we consider that each and all attributes are independent from all another, we cam multiply them for P(T|h).
7. In the end, all we have is to find the hypothesis h that maximizes the likelihood of observing P(h|T) -> P(T|ci) * P(ci)
        
- **Problem**: it does not allow for probability equal to 0.
        - Just because there is no data, does not mean it coulnd't happen in reality.
        - Solution: **Laplace Smoothing** considers k=1 for minimal time that one has seen an attribute.

## Reality of Datasets

### Potential Problems

1. One single value observed
2. Some values are missing
3. Class attributes were not observed

### Potential Solutions

1. Ignore the instances with missing values for big datasets. Law of the big numbers.
2. A **predictor is ignored** if all or most of the values are missing.
3. An **instance is ignored** if the value of the target class is missing from the training data or the value of all the predictors are missing.
4. If a case with missing some predictors' values is considered, then use the predictor(s) with non-missing values.
5. Use these non-missing values predictors to train and test the model.

### Numerical Values for Attribute Values

Two solutions:

- **Binning**: original continuous values become interval or ordinal labels.
- **Probabilistic distribution**: one density or probabilitic distribution of the attribute value (a.k.a. histogram of continuous values) to each class. Described by mean and standard deviation.

## Decision Trees

- Decision Trees are a category of methods to classify classes, i.e., it represents not only a single method of this form of Data Science.
- When the tree is built, it's by itself the model. It follows a "divide and conquer" strategy, until all of the divided partitions have only one class among them.

### Important terms

- Nodes: the test, the question of a particular attribute.
- Arrows or sides: the result from the test.
- Root node: the initial attribute to divide the dataset.
- Leafs: classes that result from the partitioning.

### Important considerations

- Specific order from root node to leafs actually matters!
    - **Then from where to start? Which would be the ideal root node?**
    - **And after the root node, from which order of attributes does one follow to build the tree?**
    
### ID3 Algorithm

Let's try to answer these last two questions with a specific algorithm of decision trees.

- It utilizes "information gain" as criteria to select the best attribute to partition.
    - It's a greedy algorithm. Takes the highest gains first and the result may not be optimal.
    - It can overfit the data and it's difficult to use on continuous data as attributes because it creates a lot of branches.
    
It works by trying to answer the question: **Which attribute better partition the dataset to "pure" sets, in other words, to sets with a more uniformity class.

> The information gain actually comes from Information Theory and the idea behind is that mixed or not pure sets needs more information to be represented or coded. So one would prefer going to less hustle (haha) and select the attribute with which the information gain decreases the most. In CS terms, one would need more bits to code a more heterogeneous partition.

#### Formula

Sure, we got it! But how to calculate it?

$$ IG(Y, X) = H(Y) - H(Y|X) $$

Let's explain this formula now.

#### Just to illustrate information content:

- If an event is certain to happen: the information content = 0.
- The information content of a polar bear to appear in Australia is huge!
- In a sequence of events, such as to decipher a string, the information of the characters just sum up.

> If information content is measured in bits, and bits are binary. Then to measure information, we just need to count the yes/no questions to define the event.

$$ N = 2^I -> I = log2(N) = log2(1/pi) = -log2(pi) $$ with I = Information and N = 1/pi with N being the number of outcomes and pi being the probability

    - The probability of each outcome in a coin toss is 0.5 because we have two possible outcomes N=2.
    - The information to code a coin toss is I = 1 bit.

Entropy
 : It is the H in the formula and represents the average information content of an event or sequence of events, measuring the effort required to form a message or to represent a message (in this case, the partition)
 
 $$ H = E(I) = ∑pi*Ii = -∑pi*log2(pi) $$
 
 **It's the sum of the information content of each event multiplied by its probability of occurence.**
 
The formula then just represents the information content decrease from the node parent Y to the node child X.

You can also Entropy for conditional events with $$ I(y|x) = -log2p(y|x) $$
And $$ H(Y|X) = -∑∑ p(x)*p(y|x)*log2p(y|x) $$ 

### Pruning

> Pruning is a method to reduce the overfitting of the decision tree model. It's the "trimming" of the decision tree, cutting off branches and sub-trees to make it less accurate to the training data and, maybe, more accurate to unseen data. It leads to less complex trees.

#### Considerations of Pruning

1. Pre-pruning sometimes necessary.
2. The construction of the tree is stopped at certain point.
3. Nodes are set to leafs even though they may not be yet pure.
4. Post-prunning or reversing splits sometimes necessary, when, for example:
    a. The partitions are sufficient small at a prior split.
    b. The instances of a subset were pure enough at a prior split result.
    c. The instances of subsets have identical attribute values

*Parameters for specific pruning*

- Size of the leaf or size to split: if the leafs already have this many instances, stop.
- Depth of the tree: if you already split the tree this many times, stop.
- Information gain: if these partitions already have information content down to this amount, stop.

*Problem*

- Choose your parameter(s) carefully not to have oversized decision tree with small threshold or to omit useful splits with large threshold. 

#### Gain Ratio and Split points

Splitting the tree a lot of times will most definitely lead to homogenous partitions that may have only a few instances. Thus, not representing the rule or the seen pattern in the real world. Useful for countinuous data as attributes values, for example, splitting point where "Temperature > 82".

**The solution may be to end the tree in a split with the highest information decrease.**

Use of the Gain Ratio for that objective (IG / Number of created branches):

$$ GR(Y, X) = I(Y, X)/ SI(Y, X) $$

### Missing Values

Sometimes the attributes' values are not known for some instances due to time consuming efforts or just not recorded.

#### Possible solutions

1. Assign the most common value of the attribute to the instance's attribute.
2. Assign the most common value of the attribute from the instances with the same class.

### More algorithms

- **ID3**: suitable for discrete values as attributes only and has no pruning.

- **C4.5**: discrete and continuous values as attributes, pruning and missing values from attributes also possible.







