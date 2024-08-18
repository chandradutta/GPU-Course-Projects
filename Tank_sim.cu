#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#define MAXDIST 2147483645

using namespace std;

__global__ void game_simulation( int *d_x, int *d_y,int roundNo, int *d_hp, int *round_hp, int *dScore, int *min_distance, int T)
{
    unsigned idx = blockIdx.x;

    int target_id = (idx + roundNo) % T;
    long long int a = d_y[target_id] - d_y[idx];
    long long int b = d_x[target_id] - d_x[idx];

    unsigned curr_id = threadIdx.x;
    long long int c = d_y[curr_id] - d_y[idx];
    long long int d = d_x[curr_id] - d_x[idx];

    min_distance[idx] = MAXDIST;
    int dist = -1;

    if (a * d == b * c && d_hp[curr_id] > 0)
    {
        if ((b != 0 && ((d_x[target_id] > d_x[idx] && d_x[curr_id] > d_x[idx]) || (d_x[target_id] < d_x[idx] && d_x[curr_id] < d_x[idx]))) ||
            (b == 0 && ((d_y[target_id] > d_y[idx] && d_y[curr_id] > d_y[idx]) || (d_y[target_id] < d_y[idx] && d_y[curr_id] < d_y[idx]))))
        {
            dist = (b != 0) ? abs(d_x[curr_id] - d_x[idx]) : abs(d_y[curr_id] - d_y[idx]);
            atomicMin(&min_distance[idx], dist);
        }
    }

    __syncthreads();

    if (d_hp[idx] > 0)
    {
        if (min_distance[idx] == dist)
        {
            atomicSub(&round_hp[curr_id], 1);
            dScore[idx] = dScore[idx] + 1;
        }
    }
}

//***********************************************

int main(int argc, char **argv)
{
    // Variable declarations
    int M, N, T, H, *xcoord, *ycoord, *score;

    FILE *inputfilepointer;

    // File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer = fopen(inputfilename, "r");

    if (inputfilepointer == NULL)
    {
        printf("input.txt file failed to open.");
        return 0;
    }

    fscanf(inputfilepointer, "%d", &M);
    fscanf(inputfilepointer, "%d", &N);
    fscanf(inputfilepointer, "%d", &T); // T is number of Tanks
    fscanf(inputfilepointer, "%d", &H); // H is the starting Health point of each Tank

    // Allocate memory on CPU
    xcoord = (int *)malloc(T * sizeof(int)); // X coordinate of each tank
    ycoord = (int *)malloc(T * sizeof(int)); // Y coordinate of each tank
    score = (int *)malloc(T * sizeof(int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for (int i = 0; i < T; i++)
    {
        fscanf(inputfilepointer, "%d", &xcoord[i]);
        fscanf(inputfilepointer, "%d", &ycoord[i]);
    }

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    int *d_x, *d_y, *min_distance;

    thrust::device_vector<int> d_hp(T, H);
    thrust::device_vector<int> round_hp(T, H);
    thrust::device_vector<int> dScore(T, 0);

    cudaMalloc(&min_distance, T * sizeof(int));

    cudaMalloc(&d_x, T * sizeof(int));
    cudaMemcpy(d_x, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_y, T * sizeof(int));
    cudaMemcpy(d_y, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);

    int roundNo = 1, cnt = T;

    while (cnt > 1)
    {
        game_simulation<<<T, T>>>( d_x, d_y,roundNo, thrust::raw_pointer_cast(d_hp.data()), thrust::raw_pointer_cast(round_hp.data()), thrust::raw_pointer_cast(dScore.data()), min_distance, T);
        roundNo++;
        thrust::copy(round_hp.begin(), round_hp.end(), d_hp.begin());
        cnt = (thrust::count_if(d_hp.begin(), d_hp.end(), thrust::placeholders::_1 > 0));
    }

    cudaMemcpy(score, thrust::raw_pointer_cast(dScore.data()), sizeof(int) * T, cudaMemcpyDeviceToHost);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, std::micro> timeTaken = end - start;

    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    // char *exectimefilename = argv[3];
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename, "w");

    for (int i = 0; i < T; i++)
    {
        fprintf(outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    // outputfilepointer = fopen(exectimefilename, "w");
    // fprintf(outputfilepointer, "%f", timeTaken.count());
    // fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}
