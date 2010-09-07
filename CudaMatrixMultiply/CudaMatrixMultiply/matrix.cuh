
#ifndef _MATRIX_CUH
#define _MATRIX_CUH

template <class T>
class Matrix;

// Matrix multiplication kernel — thread specification
template <class T>
__global__ void Kernel_Matrix_Multiply_Simple(Matrix<T> * C, Matrix<T> * A, Matrix<T> * B)
{
    int wA = A->width;
    int wB = B->width;
    int wC = C->width;
    // 2D Thread ID
    int col = threadIdx.x;
    int row = threadIdx.y;
    // Pvalue stores the Pd element that is computed by the thread
    T Pvalue = 0;
    for (int k = 0; k < wA; ++k)
    {
        T Aelement = A->data[row * wA + k];
        T Belement = B->data[k * wB + col];
        Pvalue += Aelement * Belement;
    }
    // Write the matrix to device memory each thread writes one element
    C->data[row * wC + col] = Pvalue;
}

// Matrix multiplication kernel — thread specification
template <class T>
__global__ void Kernel_Matrix_Multiply_Simple_Tile(int wTile, int hTile, Matrix<T> * C, Matrix<T> * A, Matrix<T> * B)
{
    // get column number (x).
    int tx = threadIdx.x + blockIdx.x * wTile;

    // get row number (y).
    int ty = threadIdx.y + blockIdx.y * hTile;

    int wA = A->width;
    int wB = B->width;
    int wC = C->width;

    // Bounds checking...
    if (tx >= C->width || ty >= C->height)
        return;

    // Pvalue stores the Pd element that is computed by the thread
    T Pvalue = 0;
    for (int k = 0; k < wA; ++k)
    {
        T Aelement = A->data[ty * wA + k];
        T Belement = B->data[k * wB + tx];
        Pvalue += Aelement * Belement;
    }

    // Write the matrix to device memory each thread writes one element
    C->data[ty * wC + tx] = Pvalue;
}

// Matrix multiplication kernel — thread specification
template <class T>
__global__ void Kernel_Matrix_Multiply_Fancy(int wTile, int hTile, Matrix<T> * C, Matrix<T> * A, Matrix<T> * B)
{
    // x refers to the column number; y refers to the row number.

    // Define AS and BS.  Note i is row, j is column.
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
    
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int wA = A->width;
    int wB = B->width;

    // Index of the first sub-matrix of A processed by the block.
    // Note that this starts at column 0, row hTile*by.
    int aBegin = wA * hTile * by;

    // Index of the last sub-matrix of A processed by the block.
    // We want to process the entire row of A, row number hTile*by.
    int aEnd = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A to the right of the
    // last tile.  This is just wTile columns to the right.
    int aStep = wTile;

    // A tile is a little different for matrix B.  In this situation, the notion
    // the the height and width are reverse that of A.  So it is wTile units high,
    // and hTile units wide.

    // Index of the first sub-matrix of B processed by the block.  This is the block id
    // column (x) times the width of the tile.
    int bBegin = wTile * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = wTile * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    T Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {

        // The Cuda compiler is very inadequate for variable-sized shared blocks.
        // Declare a maximum and hope it doesn't crap out.  Alternatively, I could
        // pass an additional parameter on the kernel call to allocate shared memory,
        // but it can only handle one __shared__ variable.
#define MAX_BLOCK_SIZE 30
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ T As[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ T Bs[MAX_BLOCK_SIZE][MAX_BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        AS(ty, tx) = A->data[a + wA * ty + tx];
        BS(ty, tx) = B->data[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
        for (int k = 0; k < wTile; ++k)
        {
            Csub += AS(ty, k) * BS(k, tx);
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * hTile * by + wTile * bx;
    C->data[c + wB * ty + tx] = Csub;
};


template <class T>
class Matrix
{
public:
    int width;
    int height;
    T * data;
    bool host;
    
    // Cuda doesn't like having the defining and applied occurences of the function
    // in two different files.
    // Get a matrix element
    __device__  T DGetElement(int row, int col)
    {
        return this->data[row * this->width + col];
    }

    // Set a matrix element
    __device__  void DSetElement(int row, int col, T value)
    {
        this->data[row * this->width + col] = value;
    }
    
    /*
     * Compute matrix multiplication, C = A * B
     * C        reference data, computed but preallocated
     * A        matrix A
     * B        matrix B
     */
    static bool Multiply_Host(Matrix<T> * C, Matrix<T> * A, Matrix<T> * B)
    {
        int hA = A->height;
        int wA = A->width;
        int wB = B->width;
        for (int i = 0; i < hA; ++i)
            for (int j = 0; j < wB; ++j) {
                T sum = 0;
                for (int k = 0; k < wA; ++k) {
                    T a = A->data[i * wA + k];
                    T b = B->data[k * wB + j];
                    sum += a * b;
                }
                C->data[i * wB + j] = sum;
            }
        return true;
    };

    static bool Multiply_Simple(Matrix<T> * C, Matrix<T> * A, Matrix<T> * B)
    {
        // Assert that A, B, and Product are host-space matrices.
        if (Is_In_Device_Space(A) || Is_In_Device_Space(B) || Is_In_Device_Space(C))
            return false;
        
        int wA = A->width;
        int hA = A->height;
        int wB = B->width;
        int hB = wA;
        int wC = wB;
        int hC = hA;

        // Copy host-space matrices to device-space matrices.
        Matrix * d_A = Matrix::Factory(false, wA, hA);
        if (d_A == 0)
        {
            std::cout << "Cannot allocate matrix\n";
            return false;
        }
        Matrix * d_B = Matrix::Factory(false, wB, hB);
        if (d_B == 0)
        {
            std::cout << "Cannot allocate matrix\n";
            return false;
        }
        Matrix::Copy(d_A, A);
        Matrix::Copy(d_B, B);
        Matrix * d_C = Matrix::Factory(false, wC, hC);
        if (d_C == 0)
        {
            std::cout << "Cannot allocate matrix\n";
            return false;
        }

        // Try host page locked memory if required.
        std::cout << "Simple matrix multiply\n";

        // setup execution parameters
        dim3 threads(wC, hC);
        dim3 grid(1,1);
        
        Kernel_Matrix_Multiply_Simple<T><<< grid, threads >>>(d_C, d_A, d_B);
        cudaThreadSynchronize();
        cudaError_t err = cudaGetLastError();
        if (Check_CUDA_Error(err, "kernel call error"))
            return false;
        return Matrix::Copy(C, d_C);
    };

    static bool Multiply_Simple_Tile(Matrix<T> * C, Matrix<T> * A, Matrix<T> * B, int wTile, int hTile)
    {
        // Assert that A, B, and Product are host-space matrices.
        if (Is_In_Device_Space(A) || Is_In_Device_Space(B) || Is_In_Device_Space(C))
            return false;
        
        int wA = A->width;
        int hA = A->height;
        int wB = B->width;
        int hB = wA;
        int wC = wB;
        int hC = hA;

        // Note if the tile is not sized so that an integral number fit exactly into the
        // product, return error.
        if (wC % wTile != 0)
        {
            std::cout << "Tile width " << wTile
                << " does not divide matrix width " << wC
                << " evenly.  Try a different wTile size\n";
            return false;
        }
        if (hC % hTile != 0)
        {
            std::cout << "Tile height " << hTile
                << " does not divide matrix height " << hC
                << " evenly.  Try a different hTile size\n";
            return false;
        }

        // Copy host-space matrices to device-space matrices.
        Matrix * d_A = Matrix::Factory(false, wA, hA);
        if (d_A == 0)
        {
            std::cout << "Cannot allocate matrix\n";
            return false;
        }
        Matrix * d_B = Matrix::Factory(false, wB, hB);
        if (d_B == 0)
        {
            std::cout << "Cannot allocate matrix\n";
            return false;
        }
        Matrix::Copy(d_A, A);
        Matrix::Copy(d_B, B);
        Matrix * d_C = Matrix::Factory(false, wC, hC);
        if (d_C == 0)
        {
            std::cout << "Cannot allocate matrix\n";
            return false;
        }
        std::cout << "Simple tile matrix multiply\n";
        
        // setup execution parameters
        // Divide the matrix into tiles, which is passed into this routine.
        
        dim3 threads(wTile, hTile);
        dim3 grid(wC / wTile, hC / hTile);
        
        Kernel_Matrix_Multiply_Simple_Tile<T><<< grid, threads >>>(wTile, hTile, d_C, d_A, d_B);
        cudaThreadSynchronize();
        cudaError_t err = cudaGetLastError();
        if (Check_CUDA_Error(err, "kernel call error"))
            return false;
        return Matrix::Copy(C, d_C);
    };

    static bool Multiply_Fancy_Tile(Matrix * C, Matrix * A, Matrix * B, int wTile, int hTile)
    {
        // Assert that A, B, and Product are host-space matrices.
        if (Is_In_Device_Space(A) || Is_In_Device_Space(B) || Is_In_Device_Space(C))
            return false;
        
        int wA = A->width;
        int hA = A->height;
        int wB = B->width;
        int hB = wA;
        int wC = wB;
        int hC = hA;

        // Note if the tile is not sized so that an integral number fit exactly into the
        // product, return error.
        if (wC % wTile != 0)
        {
            std::cout << "Tile width " << wTile
                << " does not divide matrix width " << wC
                << " evenly.  Try a different wTile size\n";
            return false;
        }
        if (hC % hTile != 0)
        {
            std::cout << "Tile height " << hTile
                << " does not divide matrix height " << hC
                << " evenly.  Try a different hTile size\n";
            return false;
        }

        // Copy host-space matrices to device-space matrices.
        Matrix<T> * d_A = Matrix<T>::Factory(false, wA, hA);
        if (d_A == 0)
        {
            std::cout << "Cannot allocate matrix\n";
            return false;
        }
        Matrix<T> * d_B = Matrix<T>::Factory(false, wB, hB);
        if (d_B == 0)
        {
            std::cout << "Cannot allocate matrix\n";
            return false;
        }
        Matrix<T>::Copy(d_A, A);
        Matrix<T>::Copy(d_B, B);
        Matrix<T> * d_C = Matrix<T>::Factory(false, wC, hC);
        if (d_C == 0)
        {
            std::cout << "Cannot allocate matrix\n";
            return false;
        }
        std::cout << "Fancy tile matrix multiply\n";
        
        // setup execution parameters
        // Divide the matrix into tiles, which is passed into this routine.
        
        dim3 threads(wTile, hTile);
        dim3 grid(wC / wTile, hC / hTile);

        Kernel_Matrix_Multiply_Fancy<T><<< grid, threads >>>(wTile, hTile, d_C, d_A, d_B);
        cudaThreadSynchronize();
        cudaError_t err = cudaGetLastError();
        if (Check_CUDA_Error(err, "kernel call error"))
            return false;
        return Matrix::Copy(C, d_C);
    };

    static bool Equal(Matrix<T> * A, Matrix<T> * B)
    {
        bool a_host;
        bool b_host;
        if (! Is_In_Device_Space((void*)A))
            a_host = true;
        else
            a_host = false;
        if (! Is_In_Device_Space((void*)B))
            b_host = true;
        else
            b_host = false;

        // four cases possible, give device or host space matrices.
        if (a_host && b_host)
        {
            if (!(A->width == B->width && A->height == B->height))
            {
                return false;
            }
            int size = A->width * A->height;
            for (int i = 0; i < size; ++i)
            {
                if (fabs((float)(A->data[i] - B->data[i])) > 1.0e-2f)
                {
                    std::cout << "diff at i " << i
                        << " " << A->data[i] << " " << B->data[i] << "\n";
                    return false;
                }
            }
            return true;
        }
        else if (b_host && ! a_host)
        {
            // First, get local copy of the A matrix.
            Matrix * local = (Matrix*)malloc(sizeof(Matrix));
            if (Check_CUDA_Error(cudaMemcpy((void*)local, A, sizeof(Matrix), cudaMemcpyDeviceToHost), "Memcpy"))
                return 0;
            if (!(local->width == B->width && local->height == B->height))
            {
                return false;
            }
            T * data = (T*)malloc(sizeof(T) * local->width * local->height);
            // Copy from device to host matrix product.
            if (Check_CUDA_Error(cudaMemcpy(data, local->data, sizeof(T) * local->width * local->height, cudaMemcpyDeviceToHost), "MemcpySimple"))
                return false;
            int size = local->width * local->height;
            for (int i = 0; i < size; ++i)
                if (fabs((float)(A->data[i] - B->data[i])) > 1.0e-2f)
                    return false;
            return true;
        }
        else if (a_host && ! b_host)
        {
            // First, get local copy of the A matrix.
            Matrix * local = (Matrix*)malloc(sizeof(Matrix));
            if (Check_CUDA_Error(cudaMemcpy((void*)local, B, sizeof(Matrix), cudaMemcpyDeviceToHost), "Memcpy"))
                return 0;
            if (!(local->width == A->width && local->height == A->height))
            {
                return false;
            }
            T * data = (T*)malloc(sizeof(T) * local->width * local->height);
            // Copy from device to host matrix product.
            if (Check_CUDA_Error(cudaMemcpy(data, local->data, sizeof(T) * local->width * local->height, cudaMemcpyDeviceToHost), "MemcpySimple"))
                return false;
            int size = local->width * local->height;
            for (int i = 0; i < size; ++i)
                if (fabs((float)(A->data[i] - B->data[i])) > 1.0e-2f)
                    return false;
            return true;
        }
        else
        {
            return false;
        }
    };

    static bool Copy(Matrix<T> * dst, Matrix<T> * src)
    {
        bool src_host;
        bool dst_host;
        if (! Is_In_Device_Space((void*)dst))
            dst_host = true;
        else
            dst_host = false;
        if (! Is_In_Device_Space((void*)src))
            src_host = true;
        else
            src_host = false;

        // four cases possible, give device or host space matrices.
        if (dst_host && src_host)
        {
            if (!(src->width == dst->width && src->height == dst->height))
            {
                dst->width = src->width;
                dst->height = src->height;
                free(dst->data);
                dst->data = (T*)malloc(sizeof(T) * src->width * src->height);
            }
            int size = dst->width * dst->height;
            for (int i = 0; i < size; ++i)
                dst->data[i] = src->data[i];
        }
        else if (dst_host && ! src_host)
        {
            // First, get local copy of source matrix.
            Matrix * local = (Matrix*)malloc(sizeof(Matrix));
            if (Check_CUDA_Error(cudaMemcpy((void*)local, src, sizeof(Matrix), cudaMemcpyDeviceToHost), "Memcpy"))
                return 0;
            if (!(local->width == dst->width && local->height == dst->height))
            {
                dst->width = local->width;
                dst->height = local->height;
                free(dst->data);
                dst->data = (T*)malloc(sizeof(T) * local->width * local->height);
            }
            // Copy from device to host matrix product.
            if (Check_CUDA_Error(cudaMemcpy(dst->data, local->data, sizeof(T) * local->width * local->height, cudaMemcpyDeviceToHost), "MemcpySimple"))
                return false;
        }
        else if (src_host && ! dst_host)
        {
            // First, get local copy of source matrix.
            Matrix * local = (Matrix*)malloc(sizeof(Matrix));
            if (Check_CUDA_Error(cudaMemcpy((void*)local, dst, sizeof(Matrix), cudaMemcpyDeviceToHost), "MemcpyA"))
                return 0;
            if (!(local->width == src->width && local->height == src->height))
            {
                local->width = src->width;
                local->height = src->height;
                if (local->data != 0)
                    cudaFree(local->data);
                if (Check_CUDA_Error(cudaMalloc((void**) &local->data, sizeof(T) * local->width * local->height), "Alloc"))
                    return false;
                Add_Ptr_In_Device_Space(local->data);
                if (Check_CUDA_Error(cudaMemcpy((void*)dst, local, sizeof(Matrix), cudaMemcpyDeviceToHost), "MemcpyB"))
                    return false;
            }
            if (Check_CUDA_Error(cudaMemcpy((void*)local->data, src->data, sizeof(T) * local->width * local->height, cudaMemcpyHostToDevice), "MemcpyC"))
                return false;
        }
        else
        {
            return false;
        }
        return true;
    };

    static Matrix * Factory(bool _host, int w, int h)
    {
        Matrix * m = 0;
        if (_host)
        {
            m = (Matrix*)malloc(sizeof(Matrix));
            if (m == 0)
                return 0;
            m->data = (T*)malloc(w * h * sizeof(T));
            if (m->data == 0)
                return 0;
            m->width = w;
            m->height = h;
            m->host = _host;
        }
        else
        {
            // Creating this object is kind of tricky.
            if (Check_CUDA_Error(cudaMalloc((void**) &m, sizeof(Matrix)), "Alloc"))
                return 0;
            Add_Ptr_In_Device_Space(m);
            T * da;
            if (Check_CUDA_Error(cudaMalloc((void**) &da, sizeof(T)*w*h), "Alloc"))
                return 0;
            Add_Ptr_In_Device_Space(da);
            // Auto allocated Matrix will not work because it won't be freed correctly.
            // Allocate heap-based Matrix.
            Matrix * hm = (Matrix*)malloc(sizeof(Matrix));
            hm->host = _host;
            hm->width = w;
            hm->height = h;
            hm->data = da;
            if (Check_CUDA_Error(cudaMemcpy((void*)m, hm, sizeof(Matrix), cudaMemcpyHostToDevice), "Memcpy"))
                return 0;
            free(hm);
        }
        return m;
    };

    void Random_Init()
    {
        int size = this->width * this->height;
        for (int i = 0; i < size; ++i)
            data[i] = rand() % 10;
    };

    ~Matrix()
    {
        if (Is_In_Device_Space(this))
        {
            Matrix * local = (Matrix*)malloc(sizeof(Matrix));
            if (Check_CUDA_Error(cudaMemcpy((void*)local, this, sizeof(Matrix), cudaMemcpyDeviceToHost), "MemcpyA"))
                return;
            if (local->host)
            {
                return;
            }
            if (local->data == 0)
                return;
            cudaFree(local->data);
        }
        else
        {
            free(this->data);
        }
    };

    void Print(char * name)
    {
        std::cout << "Matrix name: " << name << std::endl;
        for (int r = 0; r < this->height; r++)
        {
            for (int c = 0; c < this->width; c++)
            {
                std::cout << this->data[c + r * this->width] << " ";
            }
            std::cout << std::endl;
        }
    };
};
#endif
