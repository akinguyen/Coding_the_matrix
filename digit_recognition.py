import time # for timing
from mnist_loader import load_data
from power_svd import right_singular_vectors
from matutil import *
import random
from orthonormalization import orthonormalize
from vecutil import list2vec

## 1: () Squared Distance
def sq_dist(u, v):
    '''
    Input:
        - u, v: two Vecs with the same domain
    Output:
        - the square of the Euclidean distance between u and v
    Example:
        >>> u = list2vec([1, 2])
        >>> v = list2vec([2, 3])
        >>> sq_dist(u, v)
        2
    '''
    return (u-v)*(u-v)
    



## 2: () Nearest Neighbor
def nn(u, veclist):
    '''
    Input:
        - u: a Vec
        - veclist: a list of Vecs
    Output:
        - The index of the vector in veclist nearest to u
    Example:
        >>> from vecutil import list2vec
        >>> nn(list2vec([1,2]), [list2vec(l) for l in [[2,5],[1,3],[1.5,2]]])
        2
    '''
    sq_dist_list = [sq_dist(u,v) for v in veclist]
    return sq_dist_list.index(min(sq_dist_list))



## 3: () Nearest Neighbor Label
def nn_label(u, veclist, labels):
    '''
    Input:
        - u: a Vec
        - veclist: a list of Vecs
        - labels: a list of labels, one for each Vec in veclist
    Output:
        - the label of the vector in veclist nearest to u
    Example:
        >>> from vecutil import list2vec
        >>> u = list2vec([1,2])
        >>> veclist = [list2vec(l) for l in [[2,5],[1,3],[1.5,2]]]
        >>> labels = [0, 1, 0]
        >>> nn_label(u, veclist, labels)
        0
    '''
    return labels[nn(u,veclist)]



## 4: () Error Rate
def error_rate(guessed_labels, correct_labels):
    '''
    Input:
        - guessed_labels: a list of guessed labels
        - correct_labels: a list of true labels
    Output:
        - the fraction of guessed labels that are not equal to the corresponding correct label
    Example:
        >>> error_rate([0, 1, 0, 1, 1], [0, 1, 2, 3, 4])
        0.6
    '''
    return 1-sum([1 for i in range(len(guessed_labels)) if  not (guessed_labels[i] != correct_labels[i] ])/len(correct_labels)

#Load training and testing data
images, labels = load_data()
train_images = images[:3000]
train_labels = labels[:3000]
test_images = images[3000:3100]
test_labels = labels[3000:3100]


## 7: (Task 11.6.2) Procedure to Find Centroid
def find_centroid(veclist):
    '''
    Input:
        - veclist: a list of Vecs
    Output:
        - a Vec, the centroid of veclist
    Example:
        >>> from vecutil import list2vec
        >>> vs = [list2vec(l) for l in [[1,2,3],[2,3,4],[9,10,11]]]
        >>> find_centroid(vs)
        Vec({0, 1, 2},{0: 4.0, 1: 5.0, 2: 6.0})
    '''
    return sum(veclist)/len(veclist)


## 8: () Centroid of Training Images
centroid = find_centroid(train_images) 

## 9: () Centered Training Images
centered_train_images = [vector - centroid for vector in train_images]

## 10: () Centered Test Images
centered_test_images  = [vector - centroid for vector in test_images]

## 11: () Right Singular Vectors
''' Return list of right singular vectors '''
right_singular_vs = right_singular_vectors(centered_train_images,20)

## 12: () 10 Principal Components
M10 = coldict2mat(right_singular_vs[:10])

''' Find coordinate representation of given vector on column-orthogonal Matrix '''
def projected_representation(M,x):
    return M.transpose()*x

                 
## 13: () Predictions from Nearest Neighbor on 10 Principal Components
# From your Python interaction, copy the list of labels that nearest neighbor assigns
# to the 100 test images; paste the list here.
coordinated_rep_train_images_in_M10 = [projected_representation(M10,v) for v in train_images]
coordinated_rep_test_images_in_M10 = [projected_representation(M10,v) for v in test_images]

def guessed_images(train_images,test_images):
    guessed_labels = []
    for i in test_images:
        guessed_labels.append(nn_label(i,train_images,train_labels))
    return guessed_labels

guessed_labels_10 = guessed_images(coordinated_rep_train_images_in_M10,coordinated_rep_test_images_in_M10)


## 14: () Error Rate for Nearest Neighbor on 10 Principal Components
# In your Python interaction, find the error rate of nearest neighbor, and enter it here.

svd10_nn_error_rate = error_rate(guessed_labels_10,test_labels)


## 15: () 20 Principal Components
M20 = coldict2mat(right_singular_vs[:20])



## 16: () Predictions from Nearest Neighbor on 20 Principal Components
# From your Python interaction, copy the list of labels that nearest neighbor assigns
# to the 100 test images; paste the list here.
coordinated_rep_train_images_in_M20 = [projected_representation(M20,v) for v in train_images]
coordinated_rep_test_images_in_M20 = [projected_representation(M20,v) for v in test_images]

guessed_labels_20 = guessed_images(coordinated_rep_train_images_in_M20,coordinated_rep_test_images_in_M20)


## 17: () Error Rate for Nearest Neighbor on 20 Principal Components
# In your Python interaction, find the error rate of nearest neighbor, and enter it here.

svd20_nn_error_rate = error_rate(guessed_labels_20,test_labels)
