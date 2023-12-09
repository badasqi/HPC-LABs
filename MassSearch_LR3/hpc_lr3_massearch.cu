#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdint.h>
#include <map>
#include <stdio.h>
#include <iostream>
#include <random>
#include <iostream>
#include <sstream>
#include <string>
#include <locale>
#include <fstream>
using namespace std;


//Kernel function on GPU
__global__ void mass_search_GPU(char* N, char* H, int* R, int H_size, int N_number, int N_chars) {

    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;

    char n[2];
    for (int l = 0; l < N_chars; l++) {
        n[l] = N[index_y * N_chars + l];
    }
    for (int k = 0; k < N_chars; k++) {
        if (n[k] == H[index_x]) {
            R[index_y * H_size + index_x - k] -= 1;
        }
    }
    //__syncthreads();
}

int* mass_search_CPU(char* N, char* H, int* R, int N_number, int H_size, int N_chars) {

    for (int i = 0; i < N_number; i++) {
        for (int j = 0; j < H_size; j++) {
            string n;
            for (int l = 0; l < N_chars; l++) {
                n += N[i * N_chars + l];
            }
            for (int k = 0; k < n.length(); k++) {
                if (n[k] == H[j]) {
                    R[i * H_size + j - k] -= 1;
                }
            }
        }
    }
    return R;
}

int main()
{
    locale().global(locale());
    //Задаём параметры грида

    srand(time(NULL));
    for (int H_size = 32, N_number = 32; H_size <= 2048; H_size += 256, N_number += 64) {
        // Определяем размер буфера символов, количество и длину подстрок для поиска
        int N_chars = 2;
        printf("H_size =:%d\n", H_size);
        printf("N_number =:%d\n", N_number);
        printf("Substring length =:%d\n", N_chars);
        // Задаём параметры грида
        dim3 dimBlock;
        dimBlock.x = 32;
        dimBlock.y = 32;
        dim3 dimGrid;
        dimGrid.x = ceil(double(H_size) / double(dimBlock.x));
        dimGrid.y = ceil(double(N_number) / double(dimBlock.y));
        printf("GridDim := (%d,%d)\n", dimGrid.x, dimGrid.y);
        printf("BlockDim := (%d,%d)\n", dimBlock.x, dimBlock.y);

        char* alph = new char[256];
        // Строка из символов для генерации
        for (int i = 0; i < 256; i++) {
            alph[i] = static_cast<char>(i);
        }

        // Вывод размерности рабочей матрицы
        printf("\nWorking matrix dimension := %dx%d\n", H_size, N_number);
        char ch;

        // Выделение памяти для входно буфера, подстрок, рабочей матрицы для CPU и GPU для сравнения результатов
        char* H = new char[H_size];
        char* N = new char[N_number * N_chars];
        int* R_cpu = new int[N_number * H_size];
        int* R_gpu = new int[N_number * H_size];

        ofstream file_N("content/N/N_" + to_string(N_number) + ".txt");
        printf("\nSaving Substring for search to file content/N/N_%d(N_number):\n", N_number);
        // генерация подстрок для поиска
        for (int i = 0; i < N_number; i++) {
            for (int j = 0; j < N_chars; j++) {
                N[i * N_chars + j] = alph[rand() % 256];
                file_N << N[i * N_chars + j];
            }
            file_N << "\n";
        }

        //Создание файла для хранения входного буфера.
        printf("\nSaving input buffer  to file content/H/H_%d(H_size):\n", H_size);
        ofstream file_H("content/H/H_" + to_string(H_size) + ".txt");
        // генерация входного буфера
        for (int i = 0; i < H_size; i++) {
            H[i] = alph[rand() % 256];
            file_H << H[i];
        }

        // заполнение рабочей матрицы R
        for (int i = 0; i < N_number; i++) {
            for (int j = 0; j < H_size; j++) {
                R_cpu[i * H_size + j] = N_chars;
                R_gpu[i * H_size + j] = N_chars;
            }
        }

        // выделение памяти для рабочей матрицы на устройстве
        int R2b = H_size * N_number * sizeof(int);
        int* d_R = NULL;
        cudaError_t cuerr = cudaMalloc((void**)&d_R, R2b);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot malloc device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }
        // копируем рабочую матрицу с хоста на устройство
        cuerr = cudaMemcpy(d_R, R_gpu, R2b, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot coppy host to device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        //Substring Search on CPU

        //Установка точки старта
        clock_t st_time = clock();
        mass_search_CPU(N, H, R_cpu, N_number, H_size, N_chars);
        //Установка точки завершения работы CPU
        clock_t end_time = clock();

        // расчёт времени работы на CPU
        float CPU_time = ((double)end_time - st_time) / ((double)CLOCKS_PER_SEC);
        printf("\nThe CPU calculated for %.7f second(s)\n", CPU_time);

        // Substring search on GPU

        // Выделение памяти для буфера символов на устройстве
        int H2char = H_size * sizeof(char);

        char* d_H;
        cuerr = cudaMalloc((void**)&d_H, H2char);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot malloc device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        // копируем буфер символов на устройство
        cuerr = cudaMemcpy(d_H, H, H2char, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot coppy host to device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }
        // Выделение памяти для подстрок на устройстве
        int N2char = N_chars * N_number * sizeof(char);
        char* d_N;
        cuerr = cudaMalloc((void**)&d_N, N2char);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot malloc device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }
        // Копируем подстроки с хоста на устройство
        cuerr = cudaMemcpy(d_N, N, N2char, cudaMemcpyHostToDevice);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot coppy host to device: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        // Создание обработчиков событий
        cudaEvent_t start, stop;
        float gpuTime = 0.0;
        cuerr = cudaEventCreate(&start);
        cuerr = cudaEventCreate(&stop);
        // Установка точки старта
        cuerr = cudaEventRecord(start, 0);
        mass_search_GPU << < dimGrid, dimBlock >> > (d_N, d_H, d_R, H_size, N_number, N_chars);
        cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }
        // Установка точки завершения работы GPU
        cuerr = cudaEventRecord(stop, 0);
        // Синхронизация устройств
        cuerr = cudaDeviceSynchronize();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot synchronize CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        // расчёт времени работы на GPU
        cuerr = cudaEventElapsedTime(&gpuTime, start, stop);
        printf("The GPU calculated for %s: %.9f seconds\n", "kernel", gpuTime / 1000);

        // расчёт ускорения
        printf("\nAcceleration on GPU := %.9f\n", CPU_time / (gpuTime / 1000));

        // Копирование результата на хост
        cuerr = cudaMemcpy(R_gpu, d_R, R2b, cudaMemcpyDeviceToHost);
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot copy c array from device to host: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }
        // Сравнение результатов CPU и GPU
        bool flag = false;
        for (int i = 0; i < N_number; i++) {
            for (int j = 0; j < H_size; j++) {
                if (R_gpu[i * H_size + j] == R_cpu[i * H_size + j]) {
                    flag = true;
                }
                else {
                    flag = false;
                }
            }
        }
        if (flag) printf("\nResult CPU == Result GPU\n");
        else printf("\nResult CPU != Result GPU\n");

        //Создание файлов и запись результатов и входного буфера.

        ofstream file_R("content/R/Result_" + to_string(N_number) + "x" + to_string(H_size) + ".txt");
        // Вывод результатов поиска
        printf("\nSaving search results to file content/R/Result_%d(N_number)_%d(H_size):\n", N_number, H_size);
        for (int i = 0; i < N_number; i++) {
            for (int j = 0; j < H_size; j++) {
                if (R_gpu[i * H_size + j] == 0) {
                    file_R << "Substring number, index in input buffer - (n,k) := (" << to_string(i) << "," << to_string(j) << ")";
                    file_R << "\tSubstring := ";
                    for (int k = 0; k < N_chars; k++) {
                        file_R << N[i * N_chars + k];
                    }
                    file_R << endl;
                }
            }
        }
        printf("\n===============================================================================\n\n");

        file_R.close();
        file_H.close();
        file_N.close();

        delete[]H;
        delete[]R_cpu;
        delete[]R_gpu;
        delete[]N;

        cudaFree(d_R);
        cudaFree(d_N);
        cudaFree(d_H);

    }
    return 0;

}
