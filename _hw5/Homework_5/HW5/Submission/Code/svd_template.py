import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
def SVD(A, s, k):
    # TODO: Calculate probabilities p_i
    n,m = A.shape

    # TODO: Construct S matrix of size s by m
    S = np.zeros((s,m))

    # TODO: Calculate SS^T


    # TODO: Compute SVD for SS^T


    # TODO: Construct H matrix of size m by k
    H = np.zeros((m,k))


    # Return matrix H and top-k singular values sigma
    

def main():
    im = Image.open("baboon.tiff")
    A = np.array(im)
    H, sigma = SVD(A, 80, 60)
    k = 60


    # TO DO: Compute SVD for A and calculate optimal k-rank approximation for A.


    # TO DO: Use H to compute sub-optimal k rank approximation for A


    # To DO: Generate plots for original image, optimal k-rank and sub-optimal k rank approximation


    # TO DO: Calculate the error in terms of the Frobenius norm for both the optimal-k
    # rank produced from the SVD and for the k-rank approximation produced using
    # sub-optimal k-rank approximation for A using H.


if __name__ == "__main__":
    main()
