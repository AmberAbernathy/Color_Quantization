# Color_Quantization

  Program to perform color quantization using four k-means variants:
  1) Batch K-Means (Forgy, 1965 and Lloyd, 1982)
  2) Incremental Batch K-Means (Linde et al., 1980)
  3) Online K-Means (MacQueen, 1967)
  4) Incremental Online K-Means (Abernathy & Celebi, 2022)

  Authors: Amber Abernathy & M. Emre Celebi

  Contact email: ecelebi@uca.edu

  If you find this program useful, please cite:
  A. D. Abernathy and M. E. Celebi, 
  The Incremental Online K-Means Clustering Algorithm 
  and Its Application to Color Quantization,
  Expert Systems with Applications,
  in press, https://doi.org/10.1016/j.eswa.2022.117927, 2022.

  Compilation:
  g++ -O3 -std=c++11 -o test_km_algs test_km_algs.cpp -lm

  or
  
  g++ -Ofast -std=c++11 -o test_km_algs test_km_algs.cpp -lm

  Displaying command-line options:
  ./test_km_algs

  Notes: 
         1) The program requires C++11 or later due to its use of <chrono> (for 
	    measuring time). If your compiler does not support C++11, you should
	    measure time another way or eliminate time measurement.
	 2) This program does not use any object-oriented features of C++. 
	    Therefore, it should be straightforward to port it to C or Java.
 
