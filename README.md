# Fast Fourier Transfor_Using MPI
The aim of this project was to implement FFT in 2D to show strong scalability and weak scalability of Fat cluster and weak cluster on GCP using MPI for different sized images.
# Fast Fourier Transform 1D
The one-dimensional Fast Fourier Transform (1D-FFT) is a computational technique used to analyze signals in terms of their frequency components, converting from the time or spatial domain into the frequency domain. It leverages the discrete Fourier transform (DFT), a mathematical framework that represents a finite sequence of data points as complex numbers on the complex plane.
# Fast Fourier Transform 2D
The 2D-FFT algorithm functions by initially dividing the image or signal into rows. Subsequently, the 1D-FFT algorithm is applied to each row, producing frequency coefficients that are organized into a two-dimensional matrix. This matrix is then transposed, and the 1D-FFT algorithm is applied to each column. The outcome is a two-dimensional matrix of frequency coefficients that accurately portrays the original image or signal in the frequency domain.
