from matutil import listlist2mat, rowdict2mat,coldict2mat, mat2rowdict, mat2coldict
from mat import Mat
from eigenfaces import load_images
from vec import Vec
from image import image2display
import svd
from orthonormalization import orthonormalize

faces = load_images('/home/pi/faces/')
unclassified = load_images('/home/pi/unclassified/',n=9)

def listimage2vecdict(images):
    vec_dict = {}
    for i,pixels in images.items():
        F = listlist2mat(pixels)
        vec_dict.update({i:Vec({(x,y) for x in range(189) for y in range(166)},{(x,y):F[x,y] for x in range(189) for y in range(166)})})
    return vec_dict

def centered_vecdict(vecdict,centroid):
    return {i: vector - centroid for (i,vector) in vecdict.items()}

def centroid(vecdict):
    return sum([vector for i,vector in vecdict.items()])/len(vecdict)

def vec2image(vector):
    return [[ vector[(x,y)] for y in range((166))] for x in range((189))]

def original_vecdict(vecdict,centroid):
    return {i: vector + centroid for (i,vector) in vecdict.items()}

def find_n_space_closet_basis(A):
    U,S,V = svd.factor(A)
    return V

def projected_representation(M,x):
    return M.transpose()*x

def projection_length_square(M,x):
    return projected_representation(M,x)*projected_representation(M,x)

def project(M,x):
    return M*projected_representation(M,x)

def distance_squared(M,x):
    return (x - M*projected_representation(M,x))*(x - M*projected_representation(M,x))

def list_distance_squared(M,veclist):
    return [distance_squared(M,x) for x in veclist]

def orthonomalize_matrix(B):
    vec_dict = mat2coldict(B)
    vec_list = [j for (i,j) in vec_dict.items()]
    orthonormalize_vec = orthonormalize(vec_list)
    return (orthonormalize_vec)
    
