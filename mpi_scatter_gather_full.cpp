#include <mpi.h>
#include <stdio.h>
#include <valarray>
#include <fstream>
#include <complex>
#include <stdlib.h>
#include <iostream>
#include <map>

using namespace std;


// FFT

void fft1d(complex<double> *a, int Number) 
{
  if (Number <= 1) return;    // Base case

  // Division of input vector into even and odd 
  complex<double> even_number[Number / 2], odd_number[Number / 2];
  for (size_t c = 0; c < Number / 2; c++) 
  {
    even_number[c] = a[c * 2];
    odd_number[c] = a[c * 2 + 1];
  }

  // Compute the FFT of the even and odd -recursively
  fft1d(even_number, Number/2);
  fft1d(odd_number, Number/2);

  // Combine the results
  for (size_t s = 0; s < Number / 2; s++) 
  {
    complex<double> t = polar(1.0, -2 * M_PI * s / Number) * odd_number[s];
    a[s] = even_number[s] + t;
    a[s + Number / 2] = even_number[s] - t;
  }
}

// inverse fast Fourier transform 

void inverse_fft1d(complex<double> *a, int Number)
{
  // conjugate 
  for (int c = 0; c < Number; c++)
  {
    a[c] = conj(a[c]);
  }
  // invoking the 1D-fft.
  fft1d(a, Number);
  
  // conjugate 
  for (int c = 0; c < Number; c++)
  {
    a[c] = conj(a[c]);
  }
  
  // Normalizing the data/values. 
  for (int c = 0; c < Number; c++)
  {
    a[c] /= Number;
  }
}


int main (int arg_count, char *arg_Vector[]) 
{
  // Mapping image sizes to an integer
map<int, int> resolution{{ 512, 0 }, { 1024, 1 }, { 2048, 2 }, { 4096, 3 }};

string images[2][4] = {
        {"source_gray_image/512_gray.txt", "source_gray_image/1024_gray.txt", "source_gray_image/2048_gray.txt", "source_gray_image/4096_gray.txt"},
        {"fft_txt/512_fft.txt", "fft_txt/1024_fft.txt", "fft_txt/2048_fft.txt", "fft_txt/4096_fft.txt"}
    };

  MPI_Status state;
  int rank_value, scale; // scale shows the size to be stored
  int validsize;

  validsize = atoi(arg_Vector[1]);
  const size_t rows = validsize;
  const size_t columns = validsize;
  

  MPI_Init(&arg_count, &arg_Vector);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_value);
  MPI_Comm_size(MPI_COMM_WORLD, &scale);
  
  int chunk = validsize/scale;
  double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17;

  
  complex<double> master_store[rows][columns]; 
  complex<double> slave_store[chunk][validsize];
  
  if (rank_value == 0)
  {
    t1 = MPI_Wtime();

    // Reading the file route.
    ifstream b(images[0][resolution[validsize]]);
    string line;
    int first_dmns_count = 0;

    // Reading the txt file to put in a 2D matrix.
    while (getline(b, line)) 
    {
      int size;
      stringstream stringstream(line);
      int second_dmns_count = 0;
      while (stringstream >> size) 
      {
        master_store[first_dmns_count][second_dmns_count] = size;
        second_dmns_count++;
      }
      first_dmns_count++;
    }
    t2 = MPI_Wtime();
    printf("Time to read the image\n"); 
    cout << (t2-t1) << endl;
    printf(" *****************************************FFT Starts here ***************************************\n");
  }  

  /****************************************************************************************************/

  //Scatter Data -first time

  MPI_Scatter(master_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, slave_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  
  if (rank_value == 0)
  {
    t3 = MPI_Wtime();
    printf("Time to Scatter the data for the first time from master to all the slade nodes.\n"); 
    cout << (t3-t2) << endl;
  }

  // FFT Part starts from here.
  // row wise 1D fft .

  for (int row = 0; row < chunk; row++)
  {
    fft1d(slave_store[row], validsize);
  }
  
  if (rank_value == 0)
  {
    t4 = MPI_Wtime();
    printf("Time to do first fft1d row wise\n"); 
    cout << (t4-t3) << endl;
  }

  //Gather Data - first time
  MPI_Gather(slave_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, master_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
 
  if (rank_value == 0)
  {
    t5 = MPI_Wtime();
    printf("Time to Gather the data for the first time from all the slave nodes to the master.\n"); 
    cout << (t5-t4) << endl;
  
    // Transpose the matrix - first time.
    for (size_t i = 0; i < rows; ++i) 
    {
      for (size_t j = i + 1; j < columns; ++j) 
      {
        // Swap element at (i, j) with (j, i)
        std::swap(master_store[i][j], master_store[j][i]);
      } 
    }
    t6 = MPI_Wtime();
    printf("Time to do transpose first time\n"); 
    cout << (t6-t5) << endl;
  }

  //Scatter Data - second time
  MPI_Scatter(master_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, slave_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
   
  if (rank_value == 0)
  {
    t7 = MPI_Wtime();
    printf("Time to Scatter the data for the second time from master to all the slade nodes.\n"); 
    cout << (t7-t6) << endl;
  }

  // column wise 1D fft.
  for (int col = 0; col < chunk; col++)
  {
    fft1d(slave_store[col], validsize);
  }
  
  if (rank_value == 0)
  {
    t8 = MPI_Wtime();
    printf("Time to do second time fft1d column wise\n"); 
    cout << (t8-t7) << endl;
  }

  //Gather Data - second time
  MPI_Gather(slave_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, master_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
 
  if (rank_value == 0)
  {
    t9 = MPI_Wtime();
    printf("Time to Gather the data for the second time from all the slave nodes to the master.\n"); 
    cout << (t9-t8) << endl;
    printf(" *****************************************FFT Ends here ***************************************\n");
    printf(" *****************************************Inverse FFT Starts here ***************************************\n");
  }

  // FFT END
  /****************************************************************************************************/

  // Inverse_FFT START.
  //Scatter Data-  third time
  MPI_Scatter(master_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, slave_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

  if (rank_value == 0)
  {
    t10 = MPI_Wtime();
    printf("Time to Scatter the data for the third time from master to all the slade nodes.\n"); 
    cout << (t10-t9) << endl;
  }

  // column wise inverse_fft1D.
  for (int col = 0; col < chunk; col++)
  {
    inverse_fft1d(slave_store[col], validsize);
  }
  
  if (rank_value == 0)
  {
    t11 = MPI_Wtime();
    printf("Time to do first time inverse_fft1d column wise\n"); 
    cout << (t11-t10) << endl;
  }

  //Gather Data-  third time
  MPI_Gather(slave_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, master_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
 
  if (rank_value == 0)
  {
    t12 = MPI_Wtime();
    printf("Time to Gather the data for the third time from all the slave nodes to the master.\n"); 
    cout << (t12-t11) << endl;
    
    // Transpose  matrix -  second time.
    for (size_t i = 0; i < rows; ++i) 
    {
      for (size_t j = i + 1; j < columns; ++j) 
      {
        // Swap element at (i, j) with  (j, i)
        std::swap(master_store[i][j], master_store[j][i]);
      } 
    }
    t13 = MPI_Wtime();
    printf("Time to do transpose second time\n"); 
    cout << (t13-t12) << endl;
  }

  //Scatter Data - fourth time
  MPI_Scatter(master_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, slave_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
   
  if (rank_value == 0)
  {
    t14 = MPI_Wtime();
    printf("Time to Scatter the data for the fourth time from master to all the slade nodes.\n"); 
    cout << (t14-t13) << endl;
  }

  // column wise 1D fft.
  for (int row = 0; row < chunk; row++)
  {
    inverse_fft1d(slave_store[row], validsize);
  }
  
  if (rank_value == 0)
  {
    t15 = MPI_Wtime();
    printf("Time to do second time inverse_fft1d row wise\n"); 
    cout << (t15-t14) << endl;
  }

  //Gather Data - fourth time
  MPI_Gather(slave_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, master_store, validsize*chunk, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
  
  /****************************************************************************************************/
  // Inverse_FFT END.

  if (rank_value == 0)
  {
    t16 = MPI_Wtime();
    printf("Time to Gather the data for the fourth time from all the slave nodes to the master.\n"); 
    cout << (t16-t15) << endl;
    printf(" *****************************************Inverse FFT Ends here ***************************************\n");
    
    t17 = MPI_Wtime();
    printf("Time for running the parallel code\n"); 
    cout << (t17-t1) << endl;
  }
  
  MPI_Finalize();
  return 0;
}





