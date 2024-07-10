#include <stdio.h>
#include <valarray>
#include <complex>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <map>

using namespace std;

// 1D FFT 

void fft1d(complex<double> *a, int Number) 
{
  if (Number <= 1) return;

  // Division of input vector into even and odd 
  complex<double> even_number[Number / 2], odd_number[Number / 2];
  for (size_t c = 0; c < Number / 2; c++) 
  {
    even_number[c] = a[c * 2];
    odd_number[c] = a[c * 2 + 1];
  }

  // Compute  FFT of the even and odd - recursively
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


void inverse_fft1d(complex<double> *a, int Number)
{
  // conjugate 
  for (int c = 0; c < Number; c++)
  {
    a[c] = conj(a[c]);
  }
  fft1d(a, Number);
  
  // conjugate 
  for (int c = 0; c < Number; c++)
  {
    a[c] = conj(a[c]);
  }
  
  // Normalizing the values/data. 
  for (int c = 0; c < Number; c++)
  {
    a[c] /= Number;
  }
}

int main (int arg_count, char *arg_vector[]) 
{
  map<int, int> resolution{{ 512, 0 }, { 1024, 1 }, { 2048, 2 }, { 4096, 3 }};

  string images[2][4] = {
        {"source_gray_image/512_gray.txt", "source_gray_image/1024_gray.txt", "source_gray_image/2048_gray.txt", "source_gray_image/4096_gray.txt"},
        {"fft_txt/512_fft.txt", "fft_txt/1024_fft.txt", "fft_txt/2048_fft.txt", "fft_txt/4096_fft.txt"}
    };

  int validsize;
  validsize = atoi(arg_vector[1]);
  const size_t rows = validsize;
  const size_t columns = validsize;

  
  clock_t t1, t2, t3, t4, t5, t6, t7, t8, t9, t10;

  t1 = clock();
  complex<double> buffer[rows][columns]; 

  // Reading route for the file.
  ifstream b(images[0][resolution[validsize]]);
  
  string line;
  int first_Dmns_count = 0;

  // Reading the txt file to put in a 2d matrix.
  while (getline(b, line)) 
  {
    int size;
    stringstream stringstream(line);
    int second_dmns_count = 0;
    while (stringstream >> size) 
    {
      buffer[first_Dmns_count][second_dmns_count] = size;
      second_dmns_count++;
    }
    first_Dmns_count++;
  }

  t2 = clock();
  printf("Time to read the image\n"); 
  cout << (t2-t1)/1000000.0 << endl;
  
  // FFT Part START
  // row wise 1D fft.
  for (int row = 0; row < rows; row++)
  {
    fft1d(buffer[row], validsize);
  }
  
  t3 = clock();
  printf("Time to do first fft1d row wise\n"); 
  cout << (t3-t2)/1000000.0 << endl;

  // Transpose - first time.
  for (size_t i = 0; i < rows; ++i) 
  {
    for (size_t j = i + 1; j < columns; ++j) 
    {
      // Swap element at (i, j) with (j, i)
      std::swap(buffer[i][j], buffer[j][i]);
    } 
  }
  
  t4 = clock();
  printf("Time to do transpose first time\n"); 
  cout << (t4-t3)/1000000.0 << endl;

  // column wise 1D fft
  for (int col = 0; col < columns; col++)
  {
    fft1d(buffer[col], validsize);
  }

  t5 = clock();
  printf("Time to do second fft1d column wise\n"); 
  cout << (t5-t4)/1000000.0 << endl;
  
  // FFT END
  /********************************************************/
  // Inverse FFT START

  // row wise 1D ifft 
  for (int row = 0; row < rows; row++)
  {
    inverse_fft1d(buffer[row], validsize);
  }

  t6 = clock();
  printf("Time to do first inverse_fft1d column wise\n"); 
  cout << (t6-t5)/1000000.0 << endl;
  
  // Transpose matrix - second time.
  for (size_t i = 0; i < rows; ++i) 
  {
    for (size_t j = i + 1; j < columns; ++j) 
    {
      // Swap element at (i, j) with  (j, i)
      std::swap(buffer[i][j], buffer[j][i]);
    } 
  }

  t7 = clock();
  printf("Time to do transpose second time\n"); 
  cout << (t7-t6)/1000000.0 << endl;

  // column wise 1D inversefft 
  for (int col = 0; col < columns; col++)
  {
    inverse_fft1d(buffer[col], validsize);
  }

  t8 = clock();
  printf("Time to do second inverse_fft1d row wise\n"); 
  cout << (t8-t7)/1000000.0 << endl;
  t9 = clock();
  
  printf("Time for running the serial code\n"); 
  cout << (t9-t1)/1000000.0 << endl;
  return 0;
}
