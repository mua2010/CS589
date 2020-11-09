# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# import math
# def SVD(A, s, k):
#     # TODO: Calculate probabilities p_i
#     n,m = A.shape
#     probabilities = list()
#     counter = 0
#     while counter < n:
#         upper = math.pow(np.linalg.norm(A[counter,:], ord=2), 2)
#         lower = math.pow(np.linalg.norm(A), 2)
#         probabilities.append(upper / lower)
#         counter += 1
#     # TODO: Construct S matrix of size s by m
#     S = np.zeros((s,m))
#     counter = 0
#     while counter < s:
#         random_choice = np.random.choice(list(range(n)), p=probabilities)
#         S[counter] = A[random_choice,:]
#         counter += 1

#     # TODO: Calculate SS^T
#     ss_to_t = np.matmul(S, S.T)

#     # TODO: Compute SVD for SS^T
#     svd = np.linalg.svd(ss_to_t)
#     W = svd[0]
#     values = svd[1]

#     # TODO: Construct H matrix of size m by k
#     H = np.zeros((k, m))
#     counter = 0
#     while counter < k:
#         mat_mul = np.matmul(S.T, W[:, counter])
#         linalg_norm = np.linalg.norm(mat_mul, ord=2)
#         H[counter] = mat_mul / linalg_norm
#         counter += 1
#     values.sort()
#     values = values[::-1]
#     top_k_singular_values_sigma = values[:k]

#     # Return matrix H and top-k singular values sigma
#     return H.T, top_k_singular_values_sigma

# def main():
#     im = Image.open("../../Data/baboon.tiff")
#     A = np.array(im)
#     k = 60


#     # TO DO: Compute SVD for A and calculate optimal k-rank approximation for A.
#     linalg_svd = np.linalg.svd(A)
#     U = linalg_svd[0]
#     S = np.array(linalg_svd[1][:k])
#     Vh = linalg_svd[2]
#     lamda_matrix = np.zeros((512,512))
#     counter = 0
#     while counter < k:
#         lamda_matrix[counter][counter] = S[counter]
#         counter += 1
#     optimal = np.matmul(np.matmul(U, lamda_matrix[:, :k]), Vh[:k, :])
#     H, sigma = SVD(A, 80, 60)

#     # TO DO: Use H to compute sub-optimal k rank approximation for A
#     sub_optimal = np.matmul(np.matmul(A, H), H.T)

#     # To DO: Generate plots for original image, optimal k-rank and sub-optimal k rank approximation
#     plt.title("original Image")
#     plt.imshow(A)
#     plt.savefig('../Figures/q2.3a_orignal.png')

#     plt.title("optimal k-rank")
#     plt.imshow(optimal)
#     plt.savefig('../Figures/q2.3a_optimal_k_rank.png')

#     plt.title("sub-optimal k-rank")
#     plt.imshow(sub_optimal)
#     plt.savefig('../Figures/q2.3a_sub_optimal_k_rank.png')

#     # TO DO: Calculate the error in terms of the Frobenius norm for both the optimal-k
#     # rank produced from the SVD and for the k-rank approximation produced using
#     # sub-optimal k-rank approximation for A using H.
#     print("Errors Q2.3.b")
#     print(np.linalg.norm(optimal - A, ord='fro'))
#     # sub-optimal k-rank approximation for A using H.
#     print(np.linalg.norm(sub_optimal - A, ord='fro'))

# if __name__ == "__main__":
#     main()
