/* 
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
 */

/* Define TRACK_SSE to keep track of the SSE value in batch k-means */
/*
#define TRACK_SSE
*/

#include <chrono>
#include <iostream>
#include <math.h>
#include <string.h>
#ifdef TRACK_SSE
#include <float.h>
#endif

typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned long ulong;

typedef struct
{
 double red, green, blue;
} RGB_Pixel;

typedef struct
{
 int width, height, size;
 RGB_Pixel* data;
} RGB_Image;

typedef struct
{
 int size;
 RGB_Pixel center;
} RGB_Cluster;

/* Max. L_2^2 distance in 24-bit RGB space = 3 * 255 * 255 */
#define MAX_RGB_DIST 195075

/* Max. # colors that can be requested */
#define MAX_NUM_COLORS 65536

/* 
  Powers of two for 0, 1, ..., 16. Note that 2^16 must equal MAX_NUM_COLORS.
  If you want to quantize to more than MAX_NUM_COLORS colors, extend the POW2 
  array accordingly.
 */
int POW2[] = { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };

RGB_Image * 
read_PPM ( const char *file_name )
{
 uchar byte;
 char buff[16];
 int c, max_rgb_val, i = 0;
 FILE *fp;
 RGB_Pixel *pixel;
 RGB_Image *img;

 fp = fopen ( file_name, "rb" );
 if ( !fp )
  {
   fprintf ( stderr, "Unable to open file (%s)!\n", file_name );
   exit ( EXIT_FAILURE );
  }

 /* Read image format */
 if ( !fgets ( buff, sizeof ( buff ), fp ) )
  {
   fprintf ( stderr, "Unable to read file (%s)!\n", file_name );
   exit ( EXIT_FAILURE );
  }

 /* Check the image format to make sure that it is binary */
 if ( buff[0] != 'P' || buff[1] != '6' )
  {
   fprintf ( stderr, "Invalid image format!\n" );
   exit ( EXIT_FAILURE );
  }

 img = ( RGB_Image * ) malloc ( sizeof ( RGB_Image ) );

 /* Skip comments */
 c = getc(fp);
 while ( c == '#' )
  {
   while ( getc ( fp ) != '\n' );
   c = getc ( fp );
  }

 ungetc ( c, fp );

 /* Read image dimensions */
 if ( fscanf ( fp, "%d %d", &img->width, &img->height ) != 2 )
  {
   fprintf ( stderr, "Invalid image dimensions!\n" );
   exit ( EXIT_FAILURE );
  }

 /* Read maximum component value */
 if ( fscanf ( fp, "%d", &max_rgb_val ) != 1 )
  {
   fprintf ( stderr, "Invalid maximum R, G, B value (%d)!\n", max_rgb_val );
   exit ( EXIT_FAILURE );
  }

 /* Validate maximum component value */
 if  ( max_rgb_val != 255 )
  {
   fprintf ( stderr, "Input file is not a 24-bit image!\n" );
   exit ( EXIT_FAILURE );
  }

 while ( fgetc ( fp ) != '\n' );

 /* Allocate memory for pixel data */
 img->size = img->height * img->width;
 img->data = ( RGB_Pixel * ) malloc ( img->size * sizeof ( RGB_Pixel ) );

 /* Read pixel data */
 while ( fread ( &byte, 1, 1, fp ) && i < img->size )
  {
   pixel = &img->data[i];
   pixel->red = byte;
   fread ( &byte, 1, 1, fp );
   pixel->green = byte;
   fread ( &byte, 1, 1, fp );
   pixel->blue = byte;
   i++;
  }

 fclose ( fp );

 return img;
}

void 
write_PPM ( const RGB_Image *img, const char *file_name )
{
 uchar byte;
 FILE *fp;

 fp = fopen ( file_name, "wb" );
 if ( !fp )
  {
   fprintf ( stderr, "Unable to open file (%s)!\n", file_name );
   exit ( EXIT_FAILURE );
  }

 fprintf ( fp, "P6\n%d %d\n255\n", img->width, img->height );

 for ( int i = 0; i < img->size; i++ )
  {
   byte = ( uchar ) img->data[i].red;
   fwrite ( &byte, sizeof ( uchar ), 1, fp );
   byte = ( uchar ) img->data[i].green;
   fwrite ( &byte, sizeof ( uchar ), 1, fp );
   byte = ( uchar ) img->data[i].blue;
   fwrite ( &byte, sizeof ( uchar ), 1, fp );
  }

 fclose ( fp );
}

/*
  Function to generate two quasirandom numbers from 
  a 2D Sobol sequence. Adapted from Numerical Recipies 
  in C. Upon return, X and Y fall in [0,1).
 */

#define MAX_BIT 30

void 
sob_seq ( double *x, double *y )
{
 int j, k, l;
 ulong i, im, ipp;
 static double fac;
 static int init = 0;
 static ulong ix1, ix2;
 static ulong in, *iu[2 * MAX_BIT + 1];
 static ulong mdeg[3] = { 0, 1, 2 };
 static ulong ip[3] = { 0, 0, 1 };
 static ulong iv[2 * MAX_BIT + 1] =
 { 0, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 1, 5, 7, 7, 3, 3, 5, 15, 11, 5, 15, 13, 9 };

 if ( !init )
  {
   init = 1;
   for ( j = 1, k = 0; j <= MAX_BIT; j++, k += 2 )
    {
     iu[j] = &iv[k];
    }

   for ( k = 1; k <= 2; k++ )
    {
     for ( j = 1; j <= ( int ) mdeg[k]; j++ )
      {
       iu[j][k] <<= ( MAX_BIT - j );
      }

     for ( j = mdeg[k] + 1; j <= MAX_BIT; j++ )
      {
       ipp = ip[k];
       i = iu[j - mdeg[k]][k];
       i ^= ( i >> mdeg[k] );

       for ( l = mdeg[k] - 1; l >= 1; l-- )
        {
 	 if ( ipp & 1 )
 	  {
  	   i ^= iu[j - l][k];
	  }

	 ipp >>= 1;
	}

       iu[j][k] = i;
      }
    }

   fac = 1.0 / ( 1L << MAX_BIT );
   in = 0;
  }

 im = in;
 for ( j = 1; j <= MAX_BIT; j++ )
  {
   if ( ! ( im & 1 ) )
    {
     break;
    }

   im >>= 1;
  }

 im = (j - 1) * 2;
 *x = (ix1 ^= iv[im + 1]) * fac;
 *y = (ix2 ^= iv[im + 2]) * fac;
 in++;
}

#undef MAX_BIT

/*
  Function to determine if an integer is a power of 2:
  http://graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
*/

bool
is_pow2 ( const int x )
{ 
 uint ux = ( uint ) x;

 return ux && !( ux & ( ux - 1 ) );
}

/* 
  Online K-Means Algorithm:
  S. Thompson, M. E. Celebi, and K. H. Buck, 
  Fast Color Quantization Using MacQueen’s K-Means Algorithm, 
  Journal of Real-Time Image Processing, 
  17(5): 1609-1624, 2020.  

  Notes:
        1) LR_EXP: Learning rate exponent (must be in [0.5, 1])
        2) SAMPLE_RATE: Fraction of the input pixels (must be in (0, 1]) 
	   used during the clustering process.
	3) CLUST: When the function is called, CLUST represents the initial 
	   centers. Upon return, CLUST represents the final centers.
 */

void 
online_kmeans ( const RGB_Image *img, const int num_colors, const double lr_exp, 
		const double sample_rate, RGB_Cluster *clust )
{
 int min_dist_index;
 int old_size, new_size;
 int row_idx, col_idx;
 int num_samples;
 double sob_x, sob_y;
 double del_red, del_green, del_blue;
 double dist, min_dist;
 double learn_rate;
 RGB_Pixel rand_pixel;

 if ( lr_exp < 0.5 || lr_exp > 1. )
  {
   fprintf ( stderr, "Learning rate exponent (%g) must be in [0.5, 1]\n", lr_exp );
   exit ( EXIT_FAILURE );
  }
 else if ( sample_rate <= 0.0 || sample_rate > 1. )
  {
   fprintf ( stderr, "Sampling rate (%g) must be in (0, 1]\n", sample_rate );
   exit ( EXIT_FAILURE );
  }

 num_samples = ( int ) ( sample_rate * img->size + 0.5 ); /* round */
	
 for ( int i = 0; i < num_samples; i++ ) 
  {
   /* Sample the image quasirandomly based on a Sobol' sequence */
   sob_seq ( &sob_x, &sob_y );

   /* Find the corresponding row/column indices */
   row_idx = ( int ) ( sob_y * img->height + 0.5 ); /* round */
   if ( row_idx == img->height )
    {
     row_idx--;
    }

   col_idx = ( int ) ( sob_x * img->width + 0.5 ); /* round */
   if ( col_idx == img->width )
    {
     col_idx--;
    }

   rand_pixel = img->data[row_idx * img->width + col_idx];

   /* Find the nearest center */
   min_dist = MAX_RGB_DIST;
   min_dist_index = -INT_MAX;
   for ( int j = 0; j < num_colors; j++ ) 
    {
     del_red = clust[j].center.red - rand_pixel.red;
     del_green = clust[j].center.green - rand_pixel.green;
     del_blue = clust[j].center.blue - rand_pixel.blue;

     dist = del_red * del_red + del_green * del_green + del_blue * del_blue;
     if ( dist < min_dist ) 
      {
       min_dist = dist;
       min_dist_index = j;
      }
    }

   /* Update the size of the nearest cluster */
   old_size = clust[min_dist_index].size;
   new_size = old_size + 1;

   /* Compute the learning rate */
   learn_rate = pow ( new_size, -lr_exp );	

   /* Update the center of the nearest cluster */
   clust[min_dist_index].center.red += learn_rate * 
	                               ( rand_pixel.red - clust[min_dist_index].center.red );
   clust[min_dist_index].center.green += learn_rate * 
	                                 ( rand_pixel.green - clust[min_dist_index].center.green );
   clust[min_dist_index].center.blue += learn_rate * 
	                                ( rand_pixel.blue - clust[min_dist_index].center.blue );
   clust[min_dist_index].size = new_size;
  }
}

/* Function to compute the centroid of an image */

RGB_Cluster 
compute_centroid ( const RGB_Image *img ) 
{
 double sum_red = 0.0, sum_green = 0.0, sum_blue = 0.0;
 RGB_Pixel pixel;
 RGB_Cluster centroid;

 for (int i = 0; i < img->size; i++) 
  {
   pixel = img->data[i];
   sum_red += pixel.red;
   sum_green += pixel.green;
   sum_blue += pixel.blue;
  }

 centroid.center.red = sum_red / img->size;
 centroid.center.green = sum_green / img->size;
 centroid.center.blue = sum_blue / img->size;

 return centroid;
}

/* 
  Incremental Online K-Means Algorithm:

  A. D. Abernathy and M. E. Celebi, 
  The Incremental Online K-Means Clustering Algorithm 
  and Its Application to Color Quantization,
  Expert Systems with Applications,
  accepted for publication, 2022.
  
  Notes:
        1) NUM_COLORS must be a power of 2 (otherwise the code must be 
	   modified slightly, see Abernathy & Celebi, 2022).
        2) LR_EXP: Learning rate exponent (must be in [0.5, 1])
        3) SAMPLE_RATE: Fraction of the input pixels (must be in (0, 1]) 
	   used during the clustering process.
 */

RGB_Cluster * 
inc_online_kmeans ( const RGB_Image *img, const int num_colors, 
		    const double lr_exp, const double sample_rate )
{
 int index, num_splits;
 RGB_Pixel pixel;
 RGB_Cluster *tmp_clust, *clust;
 
 if ( !is_pow2 ( num_colors ) )
  {
   fprintf ( stderr, "Number of colors (%d) must be a power of 2!\n", num_colors );
   exit ( EXIT_FAILURE );
  }

 /* Compute log2 ( num_colors ) */
 num_splits = ( int ) ( log ( num_colors ) / log ( 2 ) + 0.5 ); /* round */
 
 tmp_clust = ( RGB_Cluster * ) malloc ( ( 2 * num_colors - 1 ) * sizeof ( RGB_Cluster ) );
 clust = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );

 /* Set first center to be the dataset centroid */
 tmp_clust[0] = compute_centroid ( img );
 tmp_clust[0].size = 0;

 for ( int t = 0; t < num_splits; t++ )
  {
   for ( int n = POW2[t] - 1; n < POW2[t + 1] - 1; n++ )
    {
     /* Split c_n into c_{2n + 1} and c_{2n + 2} */
     pixel = tmp_clust[n].center;

     /* Left child */
     index = 2 * n + 1;
     tmp_clust[index].center.red = pixel.red;
     tmp_clust[index].center.green = pixel.green;
     tmp_clust[index].center.blue = pixel.blue;
     tmp_clust[index].size = 0;
	
     /* Right child */
     index++;
     tmp_clust[index].center.red = pixel.red;
     tmp_clust[index].center.green = pixel.green;
     tmp_clust[index].center.blue = pixel.blue;
     tmp_clust[index].size = 0;
    }

   /* Refine the new centers using online k-means */
   online_kmeans ( img, POW2[t + 1], lr_exp, sample_rate, 
		   tmp_clust + POW2[t + 1] - 1 );
  }
	
 /* Last NUM_COLORS centers are the final centers */
 for ( int j = 0; j < num_colors; j++ ) 
  {
   clust[j].center.red = tmp_clust[j + num_colors - 1].center.red;
   clust[j].center.green = tmp_clust[j + num_colors - 1].center.green;
   clust[j].center.blue = tmp_clust[j + num_colors - 1].center.blue;
  }

 free ( tmp_clust );
	
 return clust;
}

/* Function to compute the Mean Squared Error of a given partition */

double 
comp_MSE ( const RGB_Image *img, const RGB_Cluster *clust, const int num_colors ) 
{
 double sse = 0.0;
 double del_red, del_green, del_blue;
 double dist, min_dist;
 RGB_Pixel pixel;

 for ( int i = 0; i < img->size; i++ )
  {
   pixel = img->data[i];

   /* Find the nearest center */
   min_dist = MAX_RGB_DIST;
   for ( int j = 0; j < num_colors; j++ ) 
    {
     del_red = clust[j].center.red - pixel.red;
     del_green = clust[j].center.green - pixel.green;
     del_blue = clust[j].center.blue - pixel.blue;

     dist = del_red * del_red + del_green * del_green + del_blue * del_blue;
     if ( dist < min_dist ) 
      {
       min_dist = dist;
      }
    }

   /* Update the SSE */
   sse += min_dist;
  }

 /* Normalize the SSE */
 return sse / img->size; 
}

/* 
  Maximin initialization method (for batch/online k-means)

  For a comprehensive survey of k-means initialization methods, see
  M. E. Celebi, H. Kingravi, and P. A. Vela, 
  A Comparative Study of Efficient Initialization Methods 
  for the K-Means Clustering Algorithm, 
  Expert Systems with Applications, 40(1): 200–210, 2013.
 */

RGB_Cluster * 
maximin ( const RGB_Image *img, const int num_colors ) 
{
 int max_dist_index;
 double del_red, del_green, del_blue;
 double dist, max_dist;
 double *d;
 RGB_Pixel pixel;
 RGB_Cluster *clust;

 clust = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );
 d = ( double * ) malloc ( img->size * sizeof ( double ) );

 /* Set first center to be the dataset centroid */
 clust[0] = compute_centroid ( img );
 clust[0].size = 0;

 /* Initialize distances to infinity */
 for ( int j = 0; j < img->size; j++ ) 
  {
   d[j] = MAX_RGB_DIST;
  }

 /* Determine the remaining centers*/
 for ( int i = 0 + 1; i < num_colors; i++ )
  {
   max_dist = -MAX_RGB_DIST;
   max_dist_index = -INT_MAX;
   for ( int j = 0; j < img->size; j++ ) 
    {
     pixel = img->data[j];

     /* Compute this pixel's distance to the previously chosen center */
     del_red = clust[i - 1].center.red - pixel.red;
     del_green = clust[i - 1].center.green - pixel.green;
     del_blue = clust[i - 1].center.blue - pixel.blue;

     dist = del_red * del_red + del_green * del_green + del_blue * del_blue;
     if ( dist < d[j] ) 
      {
       /* Update the nearest-center-distance for this pixel */
       d[j] = dist;
      }
     
     if ( max_dist < d[j] ) 
      {
       /* Update the maximum nearest-center-distance so far */
       max_dist = d[j];
       max_dist_index = j;
      }
    }

   /* Pixel with maximum distance to its nearest center is chosen as a center */
   clust[i].center = img->data[max_dist_index];
   clust[i].size = 0;
  }

 free ( d );

 return clust;
}

/*
  Batch K-Means Algorithm:

  M. E. Celebi, 
  Improving the Performance of K-Means for Color Quantization,
  Image and Vision Computing, 29(4): 260–271, 2011.
  
  Notes:
        1) MAX_ITERS: Maximum # k-means iterations allowed
	2) CLUST: When the function is called, CLUST represents the initial 
	   centers. Upon return, CLUST represents the final centers.
 */

void 
batch_kmeans ( const RGB_Image *img, const int num_colors,
	       const int max_iters, RGB_Cluster *clust )
{
 int min_dist_index, size;
 int num_iters = 0, num_changes;
 int *member;
 double del_red, del_green, del_blue;
 double dist, min_dist;
 #ifdef TRACK_SSE
 double old_obj, new_obj = DBL_MAX;
 #endif
 RGB_Pixel pixel;
 RGB_Cluster *tmp_clust;

 tmp_clust = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );
 member = ( int * ) malloc ( img->size * sizeof ( int ) );

 do
  {
   num_iters++;
   num_changes = 0;

   #ifdef TRACK_SSE
   old_obj = new_obj;
   new_obj = 0.0;
   #endif

   /* Reset the new clusters */
   for ( int j = 0; j < num_colors; j++ )
   {
    tmp_clust[j].center.red = 0.0;
    tmp_clust[j].center.green = 0.0;
    tmp_clust[j].center.blue = 0.0;
    tmp_clust[j].size = 0;
   }

   for ( int i = 0; i < img->size; i++ )
    {
     pixel = img->data[i];

     /* Find the nearest center */
     min_dist = MAX_RGB_DIST;
     min_dist_index = -INT_MAX;
     for ( int j = 0; j < num_colors; j++ )
      {
       del_red = pixel.red - clust[j].center.red;
       del_green = pixel.green - clust[j].center.green;
       del_blue = pixel.blue - clust[j].center.blue;

       dist = del_red * del_red + del_green * del_green + del_blue * del_blue;
       if ( dist < min_dist )
	{
	 min_dist = dist;
	 min_dist_index = j;
	}
      }

     #ifdef TRACK_SSE
     /* Contribution of this pixel to the SSE */
     new_obj += min_dist;
     #endif

     if ( ( num_iters == 1 ) || ( member[i] != min_dist_index ) )
      {
       /* Update the membership of the pixel */
       member[i] = min_dist_index;
       num_changes++;
      }

     /* Update the temporary center & size of the nearest cluster */
     tmp_clust[min_dist_index].center.red += pixel.red;
     tmp_clust[min_dist_index].center.green += pixel.green;
     tmp_clust[min_dist_index].center.blue += pixel.blue;
     tmp_clust[min_dist_index].size += 1;
    }

   /* Update all centers */
   for ( int j = 0; j < num_colors; j++ )
    {
     if ( ( size = tmp_clust[j].size ) != 0 )
      {
       clust[j].center.red = tmp_clust[j].center.red / size;
       clust[j].center.green = tmp_clust[j].center.green / size;
       clust[j].center.blue = tmp_clust[j].center.blue / size;
      }
    }

   #ifdef TRACK_SSE
   printf ( "Iteration %d: SSE = %g ; delta SSE = %g [# changes = %d]\n",
	    num_iters, new_obj,
	    num_iters == 1 ? 0.0 : ( old_obj - new_obj ) / old_obj,
	    num_changes );
   #endif
  } 
 while ( ( 0 < num_changes ) && ( num_iters < max_iters ) );

 free ( tmp_clust );
 free ( member );
}

/* 
  Incremental Batch K-Means Algorithm:

  Y. Linde, A. Buzo, and R. Gray,
  An Algorithm for Vector Quantizer Design, 
  IEEE Transactions on Communications, 28(1): 84-95, 1980.
  
  Note: NUM_COLORS must be a power of 2 (otherwise the code must be 
	modified slightly, see Abernathy & Celebi, 2022).
 */

RGB_Cluster * 
inc_batch_kmeans ( const RGB_Image *img, const int num_colors )
{
 int index, num_splits;
 RGB_Pixel eps, pixel;
 RGB_Cluster *tmp_clust, *clust;

 if ( !is_pow2 ( num_colors ) )
  {
   fprintf ( stderr, "Number of colors (%d) must be a power of 2!\n", num_colors );
   exit ( EXIT_FAILURE );
  }

 /* Small perturbation constant */
 eps.red = eps.green = eps.blue = 0.255;
	
 /* Compute log2 ( num_colors ) */
 num_splits = ( int ) ( log ( num_colors ) / log ( 2 ) + 0.5 ); /* round */
	
 tmp_clust = ( RGB_Cluster * ) malloc ( ( 2 * num_colors - 1 ) * sizeof ( RGB_Cluster ) );
 clust = ( RGB_Cluster * ) malloc ( num_colors * sizeof ( RGB_Cluster ) );

 /* Set first center to be the dataset centroid */
 tmp_clust[0] = compute_centroid ( img );

 for ( int t = 0; t < num_splits; t++ )
  {
   for ( int n = POW2[t] - 1; n < POW2[t + 1] - 1; n++ )
    {
     /* Split c_n into c_{2n+1} and c_{2n+2} */
     pixel = tmp_clust[n].center;

     /* Left child */
     index = 2 * n + 1;
     tmp_clust[index].center.red = pixel.red;
     tmp_clust[index].center.green = pixel.green;
     tmp_clust[index].center.blue = pixel.blue;

     /* Right child */
     index++;
     tmp_clust[index].center.red = pixel.red + eps.red;
     tmp_clust[index].center.green = pixel.green + eps.green;
     tmp_clust[index].center.blue = pixel.blue + eps.blue;
    }

   /* Refine the new centers using batch k-means */
   batch_kmeans ( img, POW2[t + 1], INT_MAX, 
		  tmp_clust + POW2[t + 1] - 1 );
  }

 /* Last NUM_COLORS centers are the final centers */
 for ( int j = 0; j < num_colors; j++ ) 
  {
   clust[j].center.red = tmp_clust[j + num_colors - 1].center.red;
   clust[j].center.green = tmp_clust[j + num_colors - 1].center.green;
   clust[j].center.blue = tmp_clust[j + num_colors - 1].center.blue;
  }

 free ( tmp_clust );

 return clust;
}

/* 
  Function to perform pixel mapping 
  Note: CLUST represents the color palette
 */

RGB_Image *
map_pixels ( const RGB_Image *in_img, const RGB_Cluster *clust, const int num_colors )
{
 int min_dist_index;
 double del_red, del_green, del_blue;
 double dist, min_dist;
 RGB_Pixel pixel;
 RGB_Image *out_img;

 /* Prepare the output image */
 out_img = ( RGB_Image * ) malloc( sizeof ( RGB_Image ) );
 out_img->width = in_img->width;
 out_img->height = in_img->height;
 out_img->size = in_img->size;
 out_img->data = ( RGB_Pixel * ) malloc ( out_img->size * sizeof ( RGB_Pixel ) );

 for ( int i = 0; i < in_img->size; i++ ) 
  {
   pixel = in_img->data[i];

   /* Find the nearest center */
   min_dist = MAX_RGB_DIST;
   min_dist_index = -INT_MAX;
   for ( int j = 0; j < num_colors; j++ ) 
    {
     del_red = clust[j].center.red - pixel.red;
     del_green = clust[j].center.green - pixel.green;
     del_blue = clust[j].center.blue - pixel.blue;

     dist = del_red * del_red + del_green * del_green + del_blue * del_blue;
     if ( dist < min_dist ) 
      {
       min_dist = dist;
       min_dist_index = j;
      }
    }

   /* Assign the output pixel */
   out_img->data[i].red = clust[min_dist_index].center.red;
   out_img->data[i].green = clust[min_dist_index].center.green;
   out_img->data[i].blue = clust[min_dist_index].center.blue;
  }

 return out_img;
}

void
free_img ( const RGB_Image *img ) 
{
 free ( img->data );
 delete ( img );
}

static void
print_usage ( char *prog_name )
{
 fprintf ( stderr, "Color Quantization Using the Incremental Online K-Means Algorithm\n\n" );
 fprintf ( stderr, "Reference: A. D. Abernathy and M. E. Celebi, The Incremental Online K-Means Clustering Algorithm and Its Application to Color Quantization, Expert Systems with Applications, accepted for publication, 2022.\n\n" ); 
 fprintf ( stderr, "Usage: %s -i <input image> -n <# colors> -a <algorithm>\n\n", prog_name );
 fprintf ( stderr, "-i <input image>: input image in binary PPM format (default = in.ppm)\n\n" ); 
 fprintf ( stderr, "-o <output image>: output image in binary PPM format (default = out_<algorithm>.ppm)\n\n" ); 
 fprintf ( stderr, "-n <# colors>: # colors in the output image (must be in [2,%d]; default = 256).\n\n",
	   MAX_NUM_COLORS ); 
 fprintf ( stderr, "-a <algorithm>: bkm: batch k-means, ibkm: incremental batch k-means, okm: online k-means, iokm: incremental online k-means (string, default = iokm).\n\n" ); 
 fprintf ( stderr, "Many image manipulation software can display/convert/process PPM images including Irfanview (http://www.irfanview.com), GIMP (http://www.gimp.org), Netpbm (http://netpbm.sourceforge.net), and ImageMagick (http://www.imagemagick.org/script/index.php).\n\n" );

 exit ( EXIT_FAILURE );
}

int 
main ( int argc, char **argv )
{
 char in_file_name[256] = "in.ppm"; /* input image file name */
 char out_file_name[256] = "!$+.ppm"; /* output image file name */
 char algo[256] = "iokm"; /* algorithm name */
 int num_colors = 256; /* # colors in the output image */
 double lr_exp = 0.5; /* learning rate exponent for OKM and IOKM */
 double mse; /* mean squared error of the quantization */
 RGB_Image *in_img, *out_img;
 RGB_Cluster *clust = NULL; 

 if ( argc == 1 )
  {
   print_usage ( argv[0] );
  }

 for ( int i = 1; i < argc; i++ )
  {
   if ( !strcmp ( argv[i], "-i" ) )
    {
     strcpy ( in_file_name, argv[++i] );
    }
   else if ( !strcmp ( argv[i], "-o" ) )
    {
     strcpy ( out_file_name, argv[++i] );
    }
   else if ( !strcmp ( argv[i], "-n" ) )
    {
     num_colors = atoi ( argv[++i] );
     if ( num_colors < 2 || num_colors > MAX_NUM_COLORS ) 
      {
       fprintf ( stderr, "# colors (%d) must be in [2, %d]!\n\n", 
		 num_colors, MAX_NUM_COLORS );
       print_usage ( argv[0] );
     }
    }
   else if ( !strcmp ( argv[i], "-a" ) )
    {
     strcpy ( algo, argv[++i] );
    }
   else
    {
     print_usage ( argv[0] );
    }
  }

 /* Read the input image */
 in_img = read_PPM ( in_file_name );

 /* Perform color quantization */
 auto start = std::chrono::high_resolution_clock::now( );

 if ( !strncmp ( algo, "bkm", 3 ) )
  {
   clust = maximin ( in_img, num_colors );
   batch_kmeans ( in_img, num_colors, INT_MAX, clust );
  }
 else if ( !strncmp ( algo, "ibkm", 4 ) )
  {
   clust = inc_batch_kmeans ( in_img, num_colors ); 
  }
 else if ( !strncmp ( algo, "okm", 3 ) )
  {
   clust = maximin ( in_img, num_colors );
   online_kmeans ( in_img, num_colors, lr_exp, 1.0, clust );
  }
 else if ( !strncmp ( algo, "iokm", 4 ) )
  {
   clust = inc_online_kmeans ( in_img, num_colors, lr_exp, 0.5 );
  }
 else
  {
   fprintf ( stderr, "Algorithm (%s) must be bkm, ibkm, okm, or iokm!\n\n", algo );
   print_usage ( argv[0] );
  }

 auto stop = std::chrono::high_resolution_clock::now( );
 auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( stop - start );

 /* Compute the MSE of the quantized image */
 mse = comp_MSE ( in_img, clust, num_colors );

 printf ( "%s algorithm: MSE = %lf [time = %d ms.]\n", 
	  algo, mse, ( int ) elapsed.count ( ) );
 
 /* Write the output image to disk */
 out_img = map_pixels ( in_img, clust, num_colors );

 if ( !strncmp ( out_file_name, "!$+.ppm", 7 ) )
  {
   sprintf ( out_file_name, "out_%s.ppm", algo );
  }

 write_PPM ( out_img, out_file_name );

 /* Free memory */
 free_img ( in_img );
 free_img ( out_img );
 free ( clust );

 return EXIT_SUCCESS;
}
