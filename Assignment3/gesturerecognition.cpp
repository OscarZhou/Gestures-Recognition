#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <set>
#include <utility>
#include <chrono>
#include <ctime>
#include <thread>
#include <unistd.h>
#include <dirent.h>
#include <fstream>
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;
using namespace chrono;
using namespace cv::ml;

//Macros for color pixels

#define Mpixel(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)]

#define pixelB(image,x,y) ( (uchar *) ( ((image).data) + (y)*((image).step) ) ) [(x) * ((image).channels())]
#define pixelG(image,x,y) ( (uchar *) ( ((image).data) + (y)*((image).step) ) ) [(x) * ((image).channels())+1]
#define pixelR(image,x,y) ( (uchar *) ( ((image).data) + (y)*((image).step) ) ) [(x) * ((image).channels())+2]


#define FEATURES 23


// written by Hongyu ZHOU(Oscar) , 16242950

/*********************************************************************************************
 * Compile with:
 * g++ -std=c++0x -o gesturerecognition -O3 gesturerecognition.cpp `pkg-config --libs --cflags opencv`
 * Execute webcam code:
 * ./gesturerecognition
 * Execute static code: for example
 * ./gesturerecognition ~/Downloads/4_A.jpg
 * Note: the gesture.xml must be in the same directory with this file (gesturerecognition.cpp)
*********************************************************************************************/


int upH=180;
int upS=255;
int upV=255;
int loH=96;
int loS=82;
int loV=100;

int marker_upH=upH;
int marker_loH=loH;
int marker_upS=upS;
int marker_loS=loS;
int marker_upV=upV;
int marker_loV=loV;


Mat frame;

Mat findMarkers(Mat& image, Mat& imageHSV);
void EllipticFourierDescriptors ( vector<Point>& contour , vector< float>& CE);

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

void gestureRecognitionWithCamera()
{
	cout<<"-----------start up the webcam!-----------"<<endl;
	VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened())
    {
        cout << "Failed to open camera" << endl;
        exit(0);
    }
    cout << "Opened camera" << endl;        
    namedWindow( "result", 1 );
    namedWindow( "Binary image", 1 );
	//set trackbar
	createTrackbar("upH", "result", &marker_upH, 255, NULL);
	setTrackbarPos("upH","result",upH);
	createTrackbar("loH", "result", &marker_loH, 255,NULL);
	setTrackbarPos("loH","result",loH);
	createTrackbar("upS", "result", &marker_upS, 255,NULL);
	setTrackbarPos("upS","result",upS);
	createTrackbar("loS", "result", &marker_loS, 255,NULL);
	setTrackbarPos("loS","result",loS);
	createTrackbar("upV", "result", &marker_upV, 255,NULL);
	setTrackbarPos("upV","result",upV);
	createTrackbar("loV", "result", &marker_loV, 255,NULL);
	setTrackbarPos("loV","result",loV);
	//
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    cap >> frame;
    printf("frame size %d %d \n",frame.rows, frame.cols);
    int key=0;
    double fps=0.0;
	float r = 0.0;
    Mat imageHSV,imagemarkers, binaryimage;
	Ptr<ANN_MLP> model;
	model = load_classifier<ANN_MLP>("gesture.xml");
	Mat sample;
	
	namedWindow( "Contour" , CV_WINDOW_AUTOSIZE );
    while (1){
        system_clock::time_point start = system_clock::now();

        cap >> frame;
        if( frame.empty() )
            break;
		imagemarkers = frame.clone();			
        binaryimage = findMarkers(imagemarkers,imageHSV);

        imshow("Binary image" , binaryimage) ;
	    vector<vector<Point> > contours ;
	    findContours ( binaryimage , contours ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE) ;

	    //drawing the largest contour
	    Mat drawing = Mat::zeros ( binaryimage.size() , CV_8UC3 ) ;		    
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
		drawContours( drawing , contours , largestcontour , color , 1 , 8) ;
		
		vector<float> CE;            
        if(contours.empty()) continue;
		EllipticFourierDescriptors ( contours [largestcontour] , CE) ;
		sample = (Mat_<float>(1, CE.size()) << CE[0],CE[1],CE[2],CE[3],CE[4],CE[5],CE[6],CE[7],CE[8],CE[9],CE[10],
			                                            CE[11],CE[12],CE[13],CE[14],CE[15],CE[16],CE[17],CE[18],CE[19],CE[20],CE[21],CE[22]);
        
		r = model->predict( sample );
		//cout << "Prediction: " << r << endl;
		std::chrono::milliseconds timespan(100); // 
        std::this_thread::sleep_for(timespan);
		
        char printit1[50], printit2[50];
        sprintf(printit1,"frame: %2.1f", fps);
        sprintf(printit2,"prediction: %f", r);
        putText(drawing, printit1, cvPoint(10,30), FONT_HERSHEY_PLAIN, 2, cvScalar(255,255,255), 2, 8);
        putText(drawing, printit2, cvPoint(10,55), FONT_HERSHEY_PLAIN, 2, cvScalar(255,255,255), 2, 8);

		imshow( "Contour" , drawing ) ;
		imshow( "result", frame );
        key=waitKey(1);
        if(key==113 || key==27) exit(0);//either esc or 'q'

        system_clock::time_point end = system_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        //fps = 1000000*10.0/seconds;
        fps = 1000000/seconds;
        //cout << "frames " << fps << " seconds " << seconds << endl;
        
    }
}


void gestureRecognitionWithImage(string filename)
{
	cout<<"-----------read static image!-----------"<<endl;
	Mat image = imread(filename, 1);
    if( image.empty() )
    {
        printf(" fail to read image \n");
        exit(0);
    }
	

    Mat imageHSV,imagemarkers;
	namedWindow( "Contour" , CV_WINDOW_AUTOSIZE );
	Ptr<ANN_MLP> model;
	model = load_classifier<ANN_MLP>("gesture.xml");
	imagemarkers = image.clone();			
    Mat binaryImage = findMarkers(imagemarkers,imageHSV);

    //imshow("Binary image" , binaryImage) ;
    vector<vector<Point> > contours ;
    findContours ( binaryImage , contours ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE) ;

    //drawing the largest contour
    Mat drawing = Mat::zeros ( imagemarkers.size() , CV_8UC3 ) ;		    
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
	drawContours( drawing , contours , largestcontour , color , 1 , 8) ;
	
	vector<float> CE;
	EllipticFourierDescriptors ( contours [largestcontour] , CE) ;
	Mat sample = (Mat_<float>(1, CE.size()) << CE[0],CE[1],CE[2],CE[3],CE[4],CE[5],CE[6],CE[7],CE[8],CE[9],CE[10],
		                                            CE[11],CE[12],CE[13],CE[14],CE[15],CE[16],CE[17],CE[18],CE[19],CE[20],CE[21],CE[22]);

	float r = model->predict( sample );
	//cout << "Prediction: " << r << endl;
	
    char printit[50];
    sprintf(printit,"prediction: %f", r);
    putText(drawing, printit, cvPoint(10,30), FONT_HERSHEY_PLAIN, 2, cvScalar(255,255,255), 2, 8);
	
	
	//imshow( "result", image );
	imshow( "Contour" , drawing ) ;
}



int main(int argc , char** argv)
{
	if(argc != 2) {

		upH=180;
		upS=255;
		upV=255;
        loH=53;
        loS=4;
        loV=40;

		marker_upH=upH;
		marker_loH=loH;
		marker_upS=upS;
		marker_loS=loS;
		marker_upV=upV;
		marker_loV=loV;

        gestureRecognitionWithCamera();
	}
	else
	{
        upH=180;
		upS=255;
		upV=255;
		loH=96;
		loS=82;
		loV=100;

		marker_upH=upH;
		marker_loH=loH;
		marker_upS=upS;
		marker_loS=loS;
		marker_upV=upV;
		marker_loV=loV;
		gestureRecognitionWithImage(argv[1]);
	}

    waitKey(0);
    return 0;
}



Mat findMarkers(Mat& image, Mat& imageHSV){

    cvtColor(image,imageHSV,CV_RGB2HSV);
    medianBlur(imageHSV, imageHSV, 3);
	GaussianBlur(imageHSV, imageHSV, Size(3,3),3,3);
	Mat binaryImage;
	binaryImage.create(image.rows, image.cols, CV_8UC1);
	for (int x=0;x<imageHSV.cols;x++){
		for (int y=0;y<imageHSV.rows;y++){
			if( pixelR(imageHSV,x,y) < marker_loV || pixelR(imageHSV,x,y) > marker_upV ||
			pixelG(imageHSV,x,y) < marker_loS || pixelG(imageHSV,x,y) > marker_upS ||
			pixelB(imageHSV,x,y) < marker_loH || pixelB(imageHSV,x,y) > marker_upH   ){
				Mpixel(binaryImage, x, y) = 0;
			}
			else {
				Mpixel(binaryImage, x, y) = 255;
			}
		}
	}
	return binaryImage;
}




void EllipticFourierDescriptors ( vector<Point>& contour , vector< float>& CE){
   
    vector<float> ax, ay, bx, by;
    int m=contour.size() ;
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
}




