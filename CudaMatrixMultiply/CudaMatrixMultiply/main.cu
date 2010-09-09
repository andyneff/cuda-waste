#include "stdafx.h"
#include "matrix.cuh"

int _tmain(int argc, _TCHAR* argv[])
{
#define BASETYPE float
    // Get device properties.
    int DevID;
    cudaDeviceProp props;
    if (Check_CUDA_Error(cudaGetDevice(&DevID), "GetDevice"))
        return 1;
    if (Check_CUDA_Error(cudaGetDeviceProperties(&props, DevID), "GetProps"))
        return 1;
    std::cout << "Device " << DevID << " (" << props.name << ") capability: "
        << props.major << "." << props.minor << std::endl;

    srand(2006);

    {
        // Start with simple matrix multiply.  The matrices involved with this must
        // be small because the simple gpu implementation cannot handle large sizes.
        int scale = 1;
        int block_size = 16;
        int hA = scale * block_size / 2;
        int wA = scale * block_size;
        int hB = wA;
        int wB = scale * block_size * 2;
        int hC = hA;
        int wC = wB;

        // allocate host memory for matrices A and B
        Matrix<BASETYPE> * A = Matrix<BASETYPE>::Factory(true, wA, hA);
        Matrix<BASETYPE> * B = Matrix<BASETYPE>::Factory(true, wB, hB);
        Matrix<BASETYPE> * C = Matrix<BASETYPE>::Factory(true, wC, hC);

        // initialize host memory
        A->Random_Init();
        B->Random_Init();

        {
            struct _timeb  t1;
            struct _timeb  t2;
            std::cout << "Starting tests...\n";
            _ftime_s(&t1);
            bool success_h = Matrix<BASETYPE>::Multiply_Host(C, A, B);
            // Fail if host problem.
            if (! success_h)
                return 1;
            _ftime(&t2);
            std::cout << (double)(t2.time - t1.time + ((double)(t2.millitm - t1.millitm))/1000) << " s.\n";
        }

        //C->Print("C Host");

        {
            struct _timeb  t3;
            struct _timeb  t4;
            _ftime_s(&t3);
            Matrix<BASETYPE> * C_gpu = Matrix<BASETYPE>::Factory(true, wC, hC);

            A->Print("A");
            B->Print("B");
            C->Print("C");

            bool success = Matrix<BASETYPE>::Multiply_Simple(C_gpu, A, B);
                        std::cout << (success ? "passed1" : "failed1") << std::endl;

            _ftime(&t4);
            std::cout << (double)(t4.time - t3.time + ((double)(t4.millitm - t3.millitm))/1000) << " s.\n";

            // Check that the results betwen the host method and the GPU method are the same.
            {
                bool passed = true;
                if (! Matrix<BASETYPE>::Equal(C, C_gpu))
                    passed = false;
                std::cout << (passed ? "passed" : "failed") << std::endl;
                C_gpu->Print("C_gpu");
            }
        }
    }

    {
        // Next, do the simple tile matrix multiply.
        int scale = 10;
        int block_size = 16;
        int hA = scale * block_size / 2;
        int wA = scale * block_size;
        int hB = wA;
        int wB = scale * block_size * 2;
        int hC = hA;
        int wC = wB;

        // allocate host memory for matrices A and B
        Matrix<BASETYPE> * A = Matrix<BASETYPE>::Factory(true, wA, hA);
        Matrix<BASETYPE> * B = Matrix<BASETYPE>::Factory(true, wB, hB);
        Matrix<BASETYPE> * C = Matrix<BASETYPE>::Factory(true, wC, hC);

        // initialize host memory
        A->Random_Init();
        B->Random_Init();

        //A->Print("A");
        //B->Print("B");

        {
            struct _timeb  t1;
            struct _timeb  t2;
            std::cout << "Starting tests...\n";
            _ftime_s(&t1);
            bool success_h = Matrix<BASETYPE>::Multiply_Host(C, A, B);
            // Fail if host problem.
            if (! success_h)
                return 1;
            _ftime(&t2);
            std::cout << (double)(t2.time - t1.time + ((double)(t2.millitm - t1.millitm))/1000) << " s.\n";
        }
        
        {
            struct _timeb  t3;
            struct _timeb  t4;
            _ftime_s(&t3);
            Matrix<BASETYPE> * C_gpu = Matrix<BASETYPE>::Factory(true, wC, hC);
            bool success = Matrix<BASETYPE>::Multiply_Simple_Tile(C_gpu, A, B, block_size/2, block_size);

            _ftime(&t4);
            std::cout << (double)(t4.time - t3.time + ((double)(t4.millitm - t3.millitm))/1000) << " s.\n";

            // Check that the results betwen the host method and the GPU method are the same.
            {
                bool passed = true;
                if (! Matrix<BASETYPE>::Equal(C, C_gpu))
                    passed = false;
                std::cout << (passed ? "passed" : "failed") << std::endl;
            }
        }

        {
            struct _timeb  t3;
            struct _timeb  t4;
            _ftime_s(&t3);
            Matrix<BASETYPE> * C_gpu = Matrix<BASETYPE>::Factory(true, wC, hC);
            bool success = Matrix<BASETYPE>::Multiply_Fancy_Tile(C_gpu, A, B, 10, 10);

            _ftime(&t4);
            std::cout << (double)(t4.time - t3.time + ((double)(t4.millitm - t3.millitm))/1000) << " s.\n";

            // Check that the results betwen the host method and the GPU method are the same.
            {
                bool passed = true;
                if (! Matrix<BASETYPE>::Equal(C, C_gpu))
                    passed = false;
                std::cout << (passed ? "passed" : "failed") << std::endl;
            }
        }
    }

    return 0;
}

