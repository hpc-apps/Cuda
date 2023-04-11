#include <iostream>

#include <cmath>

#include <IL/il.h>


int main() {

  unsigned int image;

  ilInit();

  ilGenImages(1, &image);
  ilBindImage(image);
  ilLoadImage("in.jpg");

  int width, height, bpp, format;

  width = ilGetInteger(IL_IMAGE_WIDTH);
  height = ilGetInteger(IL_IMAGE_HEIGHT); 
  bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
  format = ilGetInteger(IL_IMAGE_FORMAT);

  // Récupération des données de l'image
  unsigned char* data = ilGetData();

  // Traitement de l'image
  unsigned char* out_grey = new unsigned char[ width*height ];
  unsigned char* out_sobel = new unsigned char[width*height ];

  for( std::size_t i = 0 ; i < width*height ; ++i )
  {
    // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024
    out_grey[ i ] = ( 307 * data[ 3*i ]
		       + 604 * data[ 3*i+1 ]
		       + 113 * data[ 3*i+2 ]
		       ) >> 10;
  }
  
  unsigned int i, j, c;

  int h, v, res;


  for(j = 1 ; j < height - 1 ; ++j) {

    for(i = 1 ; i < width - 1 ; ++i) {

	// Horizontal
	h =     out_grey[((j - 1) * width + i - 1) ] -     out_grey[((j - 1) * width + i + 1) ]
	  + 2 * out_grey[( j      * width + i - 1) ] - 2 * out_grey[( j      * width + i + 1) ]
	  +     out_grey[((j + 1) * width + i - 1) ] -     out_grey[((j + 1) * width + i + 1) ];

	// Vertical

	v =     out_grey[((j - 1) * width + i - 1) ] -     out_grey[((j + 1) * width + i - 1) ]
	  + 2 * out_grey[((j - 1) * width + i    ) ] - 2 * out_grey[((j + 1) * width + i    ) ]
	  +     out_grey[((j - 1) * width + i + 1) ] -     out_grey[((j + 1) * width + i + 1) ];

	//h = h > 255 ? 255 : h;
	//v = v > 255 ? 255 : v;

	res = h*h + v*v;
	res = res > 255*255 ? res = 255*255 : res;

	out_sobel[ j * width + i ] = sqrt(res);

    }

  }

  //Placement des données dans l'image
  ilTexImage( width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_sobel );


  // Sauvegarde de l'image


  ilEnable(IL_FILE_OVERWRITE);

  ilSaveImage("out.jpg");

  ilDeleteImages(1, &image); 

  delete [] out_grey;
  delete [] out_sobel;

}
