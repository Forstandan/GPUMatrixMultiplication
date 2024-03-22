# Matrix Multiplication using CUDA 
Using CUDA threads to compute matrix multiplication in parallel. The goal is to compare the performance of different matrix multiplication methods utilizing CUDA.

- Compilation Instructions:
  Requires CUDA installed and a compiler (Visual Studio or CLion). 
  The program takes in three command line arguments: m, k, and n, which are the dimensions of the matrices. 

- Program description:
  The program runs four separate methods for calculating the product of two matrices. To calculate the product using the CPU, the program uses basicSgemm_h(). This
  method simply uses the traditional nested loop method to multiply two matrices. The next method matrixMulKernel_1thread1element() allocates 1 thread to calculate
  one element in the product. The method matrixMulKernel_1thread1row() allocates 1 thread to calculate 1 row in the resulting product and
  matrixMulKernel_1thread1column() calculates 1 column in the resulting product.
  
  ![image](https://github.com/Forstandan/GPUMatrixMultiplication/assets/114364542/8224f482-b2dd-4490-87fb-c3af8660e412)

- Project Results:
  This program demonstrates the superiority of the GPU at calculating the product of two matrices when the size of the matrices becomes large. The experiment presented
  tests the performance of all four methods given two matrices with the dimensions n x n x n. So when n = 16, matrix A, B, and C are of the dimension 16 x 16. The resulting
  run times are recorded in seconds. 
  ![image](https://github.com/Forstandan/GPUMatrixMultiplication/assets/114364542/659289ae-facf-4516-9f55-86aca69b5d69)
  ![image](https://github.com/Forstandan/GPUMatrixMultiplication/assets/114364542/7022a721-dca7-4d4d-bcd1-51f461a4dd59)
  ![image](https://github.com/Forstandan/GPUMatrixMultiplication/assets/114364542/d51e7710-1f98-4309-aed8-2c00cede2b80)
  
