#include <iostream>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <set>
#include <utility>
#include <unistd.h>
#include <dirent.h>
#include <fstream>

using namespace std;
using namespace cv;


#define Mpixel(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)]


#define pixelB(image,x,y) ( (uchar *) ( ((image).data) + (y)*((image).step) ) ) [(x) * ((image).channels())]
#define pixelG(image,x,y) ( (uchar *) ( ((image).data) + (y)*((image).step) ) ) [(x) * ((image).channels())+1]
#define pixelR(image,x,y) ( (uchar *) ( ((image).data) + (y)*((image).step) ) ) [(x) * ((image).channels())+2]

#define FEATURES 23

const int upH=180;
const int upS=255;
const int upV=255;

const int loH=91;
const int loS=0;
const int loV=58;

int marker_upH=upH;
int marker_loH=loH;
int marker_upS=upS;
int marker_loS=loS;
int marker_upV=upV;
int marker_loV=loV;

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


vector<pair<int, string>> getFiles(string cate_dir)
{
	vector<pair<int, string>> files;//存放文件名

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
        stringstream filename(ptr->d_name);
        string segment;
        vector<std::string> seglist;

        while(std::getline(filename, segment, '_'))
        {
           seglist.push_back(segment);
        }
        if(seglist.empty()) continue;


        int gestureno = char(seglist[1][0]) - 48;

        if(gestureno<=9 && gestureno>=0)
        {
            files.push_back(std::make_pair(gestureno, ptr->d_name));
        }

	}
	closedir(dir);

	return files;
}


int main( int argc , char** argv )
{
    std::ofstream outfile("Hand.data");
    string directory = argv[1];
    vector<pair<int, string>> filenames = getFiles(directory);

    vector<pair<int, string>>::iterator it;
    for( it=filenames.begin(); it!=filenames.end(); it++)
    {
        Mat image;
        //cout<<"filename="<<directory+((pair<int, string>)(*it)).second<<endl;
        image=imread(directory+((pair<int, string>)(*it)).second);
        //cout<<"size="<<image.size<<endl;


		Mat imageHSV;
    	cvtColor(image,imageHSV,CV_RGB2HSV);
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



        //imshow("Binary image" , image) ;
        vector<vector<Point> > contours ;
        findContours ( binaryImage , contours ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE) ;
        //drawing the largest contour
        Mat drawing = Mat::zeros ( image.size() , CV_8UC3 ) ;
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


        vector<float> CE;
        EllipticFourierDescriptors ( contours [largestcontour] , CE) ;
        std::string features;

        features.append(std::to_string(((pair<int, string>)(*it)).first));

        for ( int count=0; count<FEATURES && count<CE.size() ; count++)
        {
            features.append("," + std::to_string(CE[count]));

        }
        outfile<<features<<endl;
        cout<<((pair<int, string>)(*it)).first<<((pair<int, string>)(*it)).second<<endl;
    }


    outfile.close();

    //drawContours( drawing , contours , largestcontour , color , 1 , 8) ;
    //namedWindow( "Contour" , CV_WINDOW_AUTOSIZE ) ;
    //imshow( "Contour" , drawing ) ;
    waitKey (0) ;
}
