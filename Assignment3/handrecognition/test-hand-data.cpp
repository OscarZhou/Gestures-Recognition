#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <set>
#include <utility>
#include <unistd.h>
#include <dirent.h>
#include <fstream>
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;


#define Mpixel(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)]



#define pixelB(image,x,y) ( (uchar *) ( ((image).data) + (y)*((image).step) ) ) [(x) * ((image).channels())]
#define pixelG(image,x,y) ( (uchar *) ( ((image).data) + (y)*((image).step) ) ) [(x) * ((image).channels())+1]
#define pixelR(image,x,y) ( (uchar *) ( ((image).data) + (y)*((image).step) ) ) [(x) * ((image).channels())+2]

#define FEATURES 23

#define TESTDATA 1
#if TESTDATA
const int upH=180;
const int upS=255;
const int upV=255;

const int loH=50;
const int loS=94;
const int loV=100;

#else 


//trainning parameters

const int upH=180;
const int upS=255;
const int upV=255;

const int loH=91;
const int loS=0;
const int loV=58;
#endif
//
int marker_upH=upH;
int marker_loH=loH;
int marker_upS=upS;
int marker_loS=loS;
int marker_upV=upV;
int marker_loV=loV;

Mat src;
float r;

Ptr<ANN_MLP> model;

void EllipticFourierDescriptors ( vector<Point>& contour , vector< float>& CE){

    vector<float> ax, ay, bx, by;
    int m=contour.size() ;
    //int FEATURES=20;//number of CEs we are interested in computing
    float t= (2*CV_PI)/m;
    for ( int k=0;k<FEATURES;k++)
    {
        ax.push_back(0.0);
        ay.push_back(0.0);
        bx.push_back(0.0);
        by.push_back(0.0);

        for ( int i =0;i<m; i++)
        {
             ax[k]=ax[k]+contour[i].x* cos((k+1)* t *(i));
             bx[k]=bx[k]+contour[i].x* sin((k+1)* t *(i));
             ay[k]=ay[k]+contour[i].y* cos((k+1)* t *(i));
             by[k]=by[k]+contour[i].y* sin((k+1)* t *(i));
        }
        ax[k]=(ax[k])/m;
        bx[k]=(bx[k])/m;
        ay[k]=(ay[k])/m;
        by[k]=(by[k])/m;
    }
    for ( int k=0;k<FEATURES;k++)
    {
        CE.push_back( sqrt((ax[k]* ax[k]+ay[k]* ay[k]) /(ax[0] * ax[0]+ay[0] * ay[0]) )+sqrt((bx[k]* bx[k]+by[k]* by[k]) /(bx[0] * bx[0]+by[0] * by[0]) ) );
    }
    /*
    for ( int count=0; count<FEATURES && count<CE.size() ; count++)
    {
        printf("%d CE %f ax %f ay %f bx %f by%f \n" ,count,CE[count], ax[count], ay[count],bx[count],by[count]);
    }
    */
}


vector<string> getFiles(string cate_dir)
{
	vector<string> files;//存放文件名

	DIR *dir;
	struct dirent *ptr;
	char base[20];

	if ((dir=opendir(cate_dir.c_str())) == NULL)
    {
    perror("Open dir error...");
            exit(1);
    }

	while ((ptr=readdir(dir)) != NULL)
	{
		if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
		        continue;

        //sscanf(ptr->d_name, "%s-%s:%d:%d:%d:%d:%d:%d", &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8);
        

		files.push_back(ptr->d_name);

	}
	closedir(dir);

	return files;
}

template<typename T>
static Ptr<T> load_classifier(const string& filename_to_load)
{
    // load classifier from the specified file
    Ptr<T> model = StatModel::load<T>( filename_to_load );
    if( model.empty() )
        cout << "Could not read the classifier " << filename_to_load << endl;
    else
        cout << "The classifier " << filename_to_load << " is loaded.\n";

    return model;
}

void findMarkers(int value, void * object){

    Mat imageHSV;
    Mat dstImage;
	dstImage.create(src.size(),0);
    //imshow("rgb", dstImage);

    cvtColor(src,imageHSV,CV_RGB2HSV);

	medianBlur(imageHSV, imageHSV, 3);
	GaussianBlur(imageHSV, imageHSV, Size(3,3),5,5);
	Mat binaryImage;
	binaryImage.create(dstImage.rows, dstImage.cols, CV_8UC1);
	for (int x=0;x<imageHSV.cols;x++){
		for (int y=0;y<imageHSV.rows;y++){
			if( pixelR(imageHSV,x,y) < marker_loV || pixelR(imageHSV,x,y) > marker_upV ||
			pixelG(imageHSV,x,y) < marker_loS || pixelG(imageHSV,x,y) > marker_upS ||
			pixelB(imageHSV,x,y) < marker_loH || pixelB(imageHSV,x,y) > marker_upH   ){
				Mpixel(binaryImage,x,y)=0;
			}
			else {
				Mpixel(binaryImage,x,y)=255;
			}
		}
	}

//	Canny( binaryImage, binaryImage, 50, 200, 3 );
	//cvtColor ( dstImage , dstImage , CV_BGR2GRAY ) ;
	//threshold ( dstImage , dstImage , 5 , 255 , CV_THRESH_BINARY ) ;
    //imshow("Binary image" , binaryImage) ;
    vector<vector<Point> > contours ;
    findContours ( binaryImage , contours ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE) ;

    //drawing the largest contour
    Mat drawing = Mat::zeros ( dstImage.size() , CV_8UC3 ) ;		    
    Scalar color = CV_RGB( 0 , 255 ,0 ) ;
    int largestcontour =0;
    long int largestsize =0;
    for ( int i = 0; i< contours.size() ; i++ )
    {
	if ( largestsize < contours [i].size() )
	{
	    largestsize=contours [i].size() ;
	    largestcontour=i ;
	}
    }
	drawContours( drawing , contours , largestcontour , color , 1 ) ;

	vector<float> CE;
	EllipticFourierDescriptors ( contours [largestcontour] , CE) ;

	Mat sample = (Mat_<float>(1, CE.size()) << CE[0],CE[1],CE[2],CE[3],CE[4],CE[5],CE[6],CE[7],CE[8],CE[9],CE[10],
			                                    CE[11],CE[12],CE[13],CE[14],CE[15],CE[16],CE[17],CE[18],CE[19],CE[20],CE[21],CE[22]);

	r = model->predict( sample );

	
}

int main( int argc , char** argv )
{
	int counter=0, counter1=0;
    string directory = argv[1];
    vector<string> filenames = getFiles(directory);

    vector<string>::iterator it;

	model = load_classifier<ANN_MLP>("../gesture.xml");
    for( it=filenames.begin(); it!=filenames.end(); it++)
    {
#if TESTDATA
		float classno= (float)(*it)[0]-48.0;  //for test image
		
	#else	
        
        float classno = (float)(*it)[6] - 48.0;

        if(!(classno<=9 && classno>=0))
        {
            continue;
        } //for trainning image

#endif
        //cout<<"filename="<<directory+((pair<int, string>)(*it)).second<<endl;
        src=imread(directory+ *it);
		
		Mat imageHSV,srcClone;
		srcClone = src.clone();

		//namedWindow( "captured image", 0 );
		//namedWindow( "result", 1 );
		//namedWindow( "hsv", 1 );

	    findMarkers(255, &src);

		if(classno == r)
		{
			 counter1++;
		}
		else
		{
			//cout<<"gesture no="<<classno<<", r="<<r<<endl;
			cout<<*it<<"------"<<classno<<endl;
		} 
		counter++;
    }


	cout<<"successful rate:"<<counter1/(counter*1.0)<<endl;
    //drawContours( drawing , contours , largestcontour , color , 1 , 8) ;
    //namedWindow( "Contour" , CV_WINDOW_AUTOSIZE ) ;
    //imshow( "Contour" , drawing ) ;
    waitKey (0) ;
}
