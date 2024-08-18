/*
  CS 6023 Assignment 3.
  Do not make any changes to the boiler plate code or the other files in the folder.
  Use cudaFree to deallocate any memory not in usage.
  Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

__global__ void bfs_parallel(int qsize, int *gpu_level_size, int *vertices_vis, int *level_list, int *next_level_list, int *h_off, int *h_csr)
{
    unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < qsize)
    {
        int start = h_off[level_list[tid]];
        int end = h_off[level_list[tid] + 1];
        for (int i = start; i < end; i++)
        {
            vertices_vis[h_csr[i]] = 1;
            next_level_list[h_csr[i]] = 1;
            atomicAdd(gpu_level_size, 1);
        }
    }
}
__global__ void co_ordinates_updation(int **vertex_wise_list, int *h_Gx, int *h_Gy, int v, int n, int com, int mv)
{
    int node = n;
    int command = com;
    int move = mv;
    // printf("GPU");
    int *list_to_check = vertex_wise_list[node];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < v)
    {
        if (list_to_check[tid] == 1)
        {
            if (command == 0)
            {
                // atomicSub(&h_Gx[node],move);
                // printf("atomic");
                atomicSub(&h_Gx[tid], move);
            }
            else if (command == 1)
            {
                // atomicAdd(&h_Gx[node],move);
                // printf("atomic");
                atomicAdd(&h_Gx[tid], move);
            }
            else if (command == 2)
            {
                // atomicSub(&h_Gy[node],move);
                // printf("atomic");

                atomicSub(&h_Gy[tid], move);
            }
            else
            {
                // atomicAdd(&h_Gy[node],move);
                // printf("atomic");
                atomicAdd(&h_Gy[tid], move);
            }
        }
    }
}
__global__ void generate(int *dFinalPng, int **dMesh, int *sceneOpacity, int *dOpacity, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dFrameSizeX, int *dFrameSizeY, int V, int frameSizeX, int frameSizeY, int *lock)
{
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < V)
    {
        int m = dFrameSizeX[tid];
        int n = dFrameSizeY[tid];
        int x = dGlobalCoordinatesX[tid];
        int y = dGlobalCoordinatesY[tid];
        int currSceneOpacity = dOpacity[tid];
        int *mesh = dMesh[tid];

        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int globalXpos = x + i;
                int globalYpos = y + j;
                if (globalXpos >= 0 && globalXpos < frameSizeX && globalYpos >= 0 && globalYpos < frameSizeY)
                {
                    int idx = globalXpos * frameSizeY + globalYpos;
                    while (atomicCAS(&lock[idx], 0, 1) != 0)
                        ;
                    if (currSceneOpacity > sceneOpacity[idx])
                    {
                        sceneOpacity[idx] = currSceneOpacity;
                        dFinalPng[idx] = mesh[i * n + j];
                    }
                    atomicExch(&lock[idx], 0);
                }
            }
        }
    }
}

// __global__ void generation(int*Final_output,int frameSizeX,int frameSizeY, int *hGlobalCoordinatesX, int *hGlobalCoordinatesY, int v,int* Final_opacity,int* h_op,int*Final_lock,int*hFrameSizeX,int*hFrameSizeY,int** h_mesh){
//   //  printf("enter");
//    int id=blockIdx.x*blockDim.x+threadIdx.x;
//    int g_x=hGlobalCoordinatesX[id];
//    int g_y=hGlobalCoordinatesY[id];
//   //  printf("%d",g_y);
//    int startX=hGlobalCoordinatesX[id];
//    int startY=hGlobalCoordinatesY[id];
//    int size_y=frameSizeY;
//   //  printf("h");
//    if(hGlobalCoordinatesX[id]<0){
//         startX=-(hGlobalCoordinatesX[id]);
//         g_x=0;
//     }
//    if(hGlobalCoordinatesY[id]<0){
//         startY=-(hGlobalCoordinatesY[id]);
//         g_y=0;
//     }

//    int endX=hFrameSizeX[id];
//    int endY=hFrameSizeY[id];
//    int change_gx=g_x;
//    int change_gy=g_y;
//    if (id < v)
//     {
//         for(int i=startX;i<endX && i<frameSizeX;i++){
//           for(int j=startY;i<endY && j<frameSizeY;j++){
//             int lock_old=0;
//             do{
//               lock_old = atomicCAS(&Final_lock[g_x*size_y+g_y],0,1);
//               if( lock_old == 0){
//                     int temp_opacity = Final_opacity[g_x*size_y+g_y];
//                     if( temp_opacity <= h_op[id]){
//                         Final_opacity[g_x*size_y+g_y] = h_op[id];
//                         int location=i*endY+j;
//                         Final_output[g_x*size_y+g_y] = h_mesh[id][location];
//                     }
//                     Final_lock[g_x*size_y+g_y] = 0;
//                 }
//             }while(lock_old != 0);
//             g_y++;
//           }
//           g_y=change_gy;
//           g_x++;
//         }
//     }
// }
void readFile(const char *fileName, std::vector<SceneNode *> &scenes, std::vector<std::vector<int>> &edges, std::vector<std::vector<int>> &translations, int &frameSizeX, int &frameSizeY)
{
    /* Function for parsing input file*/

    FILE *inputFile = NULL;
    // Read the file for input.
    if ((inputFile = fopen(fileName, "r")) == NULL)
    {
        printf("Failed at opening the file %s\n", fileName);
        return;
    }

    // Input the header information.
    int numMeshes;
    fscanf(inputFile, "%d", &numMeshes);
    fscanf(inputFile, "%d %d", &frameSizeX, &frameSizeY);

    // Input all meshes and store them inside a vector.
    int meshX, meshY;
    int globalPositionX, globalPositionY; // top left corner of the matrix.
    int opacity;
    int *currMesh;
    for (int i = 0; i < numMeshes; i++)
    {
        fscanf(inputFile, "%d %d", &meshX, &meshY);
        fscanf(inputFile, "%d %d", &globalPositionX, &globalPositionY);
        fscanf(inputFile, "%d", &opacity);
        currMesh = (int *)malloc(sizeof(int) * meshX * meshY);
        for (int j = 0; j < meshX; j++)
        {
            for (int k = 0; k < meshY; k++)
            {
                fscanf(inputFile, "%d", &currMesh[j * meshY + k]);
            }
        }
        // Create a Scene out of the mesh.
        SceneNode *scene = new SceneNode(i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity);
        scenes.push_back(scene);
    }

    // Input all relations and store them in edges.
    int relations;
    fscanf(inputFile, "%d", &relations);
    int u, v;
    for (int i = 0; i < relations; i++)
    {
        fscanf(inputFile, "%d %d", &u, &v);
        edges.push_back({u, v});
    }

    // Input all translations.
    int numTranslations;
    fscanf(inputFile, "%d", &numTranslations);
    std::vector<int> command(3, 0);
    for (int i = 0; i < numTranslations; i++)
    {
        fscanf(inputFile, "%d %d %d", &command[0], &command[1], &command[2]);
        translations.push_back(command);
    }
}

void writeFile(const char *outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY)
{
    /* Function for writing the final png into a file.*/
    FILE *outputFile = NULL;
    if ((outputFile = fopen(outputFileName, "w")) == NULL)
    {
        printf("Failed while opening output file\n");
    }

    for (int i = 0; i < frameSizeX; i++)
    {
        for (int j = 0; j < frameSizeY; j++)
        {
            fprintf(outputFile, "%d ", hFinalPng[i * frameSizeY + j]);
        }
        fprintf(outputFile, "\n");
    }
}

int main(int argc, char **argv)
{

    // Read the scenes into memory from File.
    const char *inputFileName = argv[1];
    int *hFinalPng;

    int frameSizeX, frameSizeY;
    std::vector<SceneNode *> scenes;
    std::vector<std::vector<int>> edges;
    std::vector<std::vector<int>> translations;
    readFile(inputFileName, scenes, edges, translations, frameSizeX, frameSizeY);
    hFinalPng = (int *)malloc(sizeof(int) * frameSizeX * frameSizeY);

    // Make the scene graph from the matrices.
    Renderer *scene = new Renderer(scenes, edges);

    // Basic information.
    int V = scenes.size();
    int E = edges.size();
    int numTranslations = translations.size();

    // Convert the scene graph into a csr.
    scene->make_csr(); // Returns the Compressed Sparse Row representation for the graph.
    int *hOffset = scene->get_h_offset();
    int *hCsr = scene->get_h_csr();
    int *hOpacity = scene->get_opacity();                      // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
    int **hMesh = scene->get_mesh_csr();                       // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
    int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX(); // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
    int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY(); // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
    int *hFrameSizeX = scene->getFrameSizeX();                 // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
    int *hFrameSizeY = scene->getFrameSizeY();                 // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

    auto start = std::chrono::high_resolution_clock::now();

    // Code begins here.
    // Do not change anything above this comment.
    // printf("hello");
    // fflush(stdout);

    // for(int i=0;i<V;i++){
    //   printf("%d",*hMesh[i]);
    // }
    int *h_off;
    int *h_csr;
    int *h_op;
    int *h_Gx;
    int *h_Gy;
    int *h_Fx;
    int *h_Fy;

    cudaMalloc(&h_off, sizeof(int) * (V + 1));
    cudaMemcpy(h_off, hOffset, sizeof(int) * (V + 1), cudaMemcpyHostToDevice);
    // sizeof (int) * this->num_nodes
    cudaMalloc(&h_op, sizeof(int) * V);
    cudaMemcpy(h_op, hOpacity, sizeof(int) * V, cudaMemcpyHostToDevice);

    cudaMalloc(&h_Gx, sizeof(int) * V);
    cudaMemcpy(h_Gx, hGlobalCoordinatesX, sizeof(int) * V, cudaMemcpyHostToDevice);

    cudaMalloc(&h_Gy, sizeof(int) * V);
    cudaMemcpy(h_Gy, hGlobalCoordinatesY, sizeof(int) * V, cudaMemcpyHostToDevice);

    cudaMalloc(&h_Fx, sizeof(int) * V);
    cudaMemcpy(h_Fx, hFrameSizeX, sizeof(int) * V, cudaMemcpyHostToDevice);

    cudaMalloc(&h_Fy, sizeof(int) * V);
    cudaMemcpy(h_Fy, hFrameSizeY, sizeof(int) * V, cudaMemcpyHostToDevice);

    cudaMalloc(&h_csr, sizeof(int) * E);
    cudaMemcpy(h_csr, hCsr, sizeof(int) * E, cudaMemcpyHostToDevice);
    // printf("hello");

    // for(int i=0;i<V;i++){
    //   printf("%d",hFrameSizeY[i]);
    // }

    int **h_mesh;
    cudaMalloc(&h_mesh, sizeof(int *) * V);

    for (int i = 0; i < V; i++)
    {
        // cudaMalloc(&(h_mesh+i),5*sizeof(int))
        int *temp;
        cudaMalloc(&temp, hFrameSizeX[i] * hFrameSizeY[i] * sizeof(int));
        cudaMemcpy(temp, hMesh[i], sizeof(int) * hFrameSizeX[i] * hFrameSizeY[i], cudaMemcpyHostToDevice);
        cudaMemcpy(&h_mesh[i], &temp, sizeof(int *), cudaMemcpyHostToDevice);
    }
    int *zeros = (int *)malloc(sizeof(int) * (V));
    for (int i = 0; i < V; i++)
    {
        zeros[i] = 0;
    }
    int *visited;
    visited = (int *)malloc(V * sizeof(int));
    int **vertex_wise_list;
    cudaMalloc(&vertex_wise_list, sizeof(int *) * V);
    for (int i = 0; i < V; i++)
    {
        int *temp;
        cudaMalloc(&temp, sizeof(int) * V);
        cudaMemcpy(temp, zeros, sizeof(int) * V, cudaMemcpyHostToDevice);
        cudaMemcpy(&vertex_wise_list[i], temp, sizeof(int *), cudaMemcpyHostToDevice);
    }
    // int* gpu_visited;
    // cudaMalloc(&gpu_visited,V*sizeof(int));
    cudaMemcpy(visited, zeros, V * sizeof(int), cudaMemcpyHostToHost);
    // for (int i = 0; i < V; i++) {
    //       printf("%d ", visited[i]);
    //   }
    // printf("hi");
    int blocks = ceil(V / 1024.0);
    for (int i = 0; i < numTranslations; i++)
    {
        int ver = translations[i][0];
        if (!visited[ver])
        {
            int *vertices_vis;
            cudaMalloc(&vertices_vis, sizeof(int) * V);
            zeros[ver] = 1;
            cudaMemcpy(vertices_vis, zeros, sizeof(int) * V, cudaMemcpyHostToDevice);
            zeros[ver] = 0;

            int *level_list;
            cudaMalloc(&level_list, sizeof(int) * V);
            zeros[0] = ver;
            cudaMemcpy(level_list, zeros, sizeof(int) * V, cudaMemcpyHostToDevice);
            zeros[0] = 0;

            int level_size = 1;
            int *gpu_level_size;
            cudaMalloc(&gpu_level_size, sizeof(int));

            visited[ver] = 1;

            int a = 0;
            do
            {
                int *next_level_list;
                cudaMalloc(&next_level_list, sizeof(int) * V);
                cudaMemset(next_level_list, 0, sizeof(int) * V);

                cudaMemcpy(gpu_level_size, &a, sizeof(int), cudaMemcpyHostToDevice);
                bfs_parallel<<<blocks, 1024>>>(level_size, gpu_level_size, vertices_vis, level_list, next_level_list, h_off, h_csr);
                cudaDeviceSynchronize();
                int *cpu_level_list = new int[V];
                int *send = new int[V];
                cudaMemcpy(cpu_level_list, next_level_list, sizeof(int) * V, cudaMemcpyDeviceToHost);
                int j = 0;
                for (int i = 0; i < V; i++)
                    if (cpu_level_list[i] == 1)
                        send[j++] = i;

                cudaMemcpy(&level_size, gpu_level_size, sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(level_list, send, sizeof(int) * level_size, cudaMemcpyHostToDevice);
            } while (level_size != 0);

            int *cpu1 = (int *)malloc(sizeof(int) * V);
            cudaMemcpy(cpu1, vertices_vis, sizeof(int) * V, cudaMemcpyDeviceToHost);
            // for (int j = 0; j < V; j++)
            // {
            //     if (cpu1[j] == 1)
            //         printf("%d ", j);
            // }
            // printf("\n");
            cudaMemcpy(&vertex_wise_list[ver], &vertices_vis, sizeof(int *), cudaMemcpyHostToDevice);
        }
    }
    int block_size = 1024;
    int blks = ceil(float(V + block_size - 1) / float(block_size));
    for (int i = 0; i < numTranslations; i++)
    {
        int node = translations[i][0];
        int command = translations[i][1];
        int move = translations[i][2];
        co_ordinates_updation<<<blks, block_size>>>(vertex_wise_list, h_Gx, h_Gy, V, node, command, move);
    }

    cudaMemcpy(hGlobalCoordinatesX, h_Gx, sizeof(int) * V, cudaMemcpyDeviceToHost);
    cudaMemcpy(hGlobalCoordinatesY, h_Gy, sizeof(int) * V, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < V; i++)
    // {
    // printf("%d, %d /n ;", hGlobalCoordinatesX[i], hGlobalCoordinatesY[i]);
    // }

    int *Final_output;
    int *Final_opacity;
    int *Final_lock;
    cudaMalloc(&Final_output, sizeof(int) * frameSizeX * frameSizeY);
    cudaMalloc(&Final_opacity, sizeof(int) * frameSizeX * frameSizeY);
    cudaMalloc(&Final_lock, sizeof(int) * frameSizeX * frameSizeY);
    cudaMemset(Final_output, 0, sizeof(int) * frameSizeX * frameSizeY);
    cudaMemset(Final_opacity, 0, sizeof(int) * frameSizeX * frameSizeY);
    cudaMemset(Final_lock, 0, sizeof(int) * frameSizeX * frameSizeY);

    int Fblocksize = 1024;
    int Fblocks = ceil(V / 1024.0);
    //  printf("h");
    // int*Final_output,int frameSizeX,int frameSizeY, int *hGlobalCoordinatesX, int *hGlobalCoordinatesX, int v,int* Final_opacity,int* h_op,int*Final_lock,int*hFrameSizeX,int*hFrameSizeY,int** h_mesh
    // int *dFinalPng, int **dMesh, int *sceneOpacity, int *dOpacity, int *dGlobalCoordinatesX, int *dGlobalCoordinatesY, int *dFrameSizeX, int *dFrameSizeY, int V, int frameSizeX, int frameSizeY, int *lock
    generate<<<Fblocks, Fblocksize>>>(Final_output, h_mesh, Final_opacity, h_op, h_Gx, h_Gy, h_Fx, h_Fy, V, frameSizeX, frameSizeY, Final_lock);
    cudaMemcpy(hFinalPng, Final_output, sizeof(int) * frameSizeX * frameSizeY, cudaMemcpyDeviceToHost);
    // Do not change anything below this comment.
    // Code ends here.

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> timeTaken = end - start;

    printf("execution time : %f\n", timeTaken.count());
    // Write output matrix to file.
    const char *outputFileName = argv[2];
    writeFile(outputFileName, hFinalPng, frameSizeX, frameSizeY);
}
