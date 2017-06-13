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

#define MpixelB(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)*((image).channels())]
#define MpixelG(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)*((image).channels())+1]
#define MpixelR(image, x, y) ((uchar *)(((image).data)+(y)*((image).step)))[(x)*((image).channels())+2]


#define FEATURES 20

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

    for ( int count=0; count<FEATURES && count<CE.size() ; count++)
    {
        printf("%d CE %f ax %f ay %f bx %f by%f \n" ,count,CE[count], ax[count], ay[count],bx[count],by[count]);
    }

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
        int gestureno = (char)ptr->d_name[6] - 48;
        if(gestureno<=9 && gestureno>=0)
        {
            files.push_back(std::make_pair(gestureno, ptr->d_name));
        }

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

int main( int argc , char** argv )
{
    Mat image;
    image=imread(argv[1]);
    //cout<<"size="<<image.size<<endl;
    cvtColor ( image , image , CV_BGR2GRAY ) ;
    threshold ( image , image , 5 , 255 , CV_THRESH_BINARY ) ;
    //imshow("Binary image" , image) ;
    vector<vector<Point> > contours ;
    findContours ( image , contours ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE) ;
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


    cout<<"filename="<<argv[1]<<endl;

    Ptr<ANN_MLP> model;
    cout<<"xml file name ="<<argv[2]<<endl;
    model = load_classifier<ANN_MLP>(argv[2]);

    Mat sample1 = (Mat_<float>(1, CE.size()) << CE[0],CE[1],CE[2],CE[3],CE[4],CE[5],CE[6],CE[7],CE[8],CE[9],CE[10],
                                                    CE[11],CE[12],CE[13],CE[14],CE[15],CE[16],CE[17],CE[18],CE[19]);

    float r = model->predict( sample1 );
    cout << "Prediction: " << r << endl;




    /*
    std::ofstream outfile("Hand.data");
    string directory = argv[1];
    vector<pair<int, string>> filenames = getFiles(directory);

    vector<pair<int, string>>::iterator it;
    for( it=filenames.begin(); it!=filenames.end(); it++)
    {
        Mat image;
        cout<<"filename="<<directory+((pair<int, string>)(*it)).second<<endl;
        image=imread(directory+((pair<int, string>)(*it)).second);
        //cout<<"size="<<image.size<<endl;
        cvtColor ( image , image , CV_BGR2GRAY ) ;
        threshold ( image , image , 5 , 255 , CV_THRESH_BINARY ) ;
        //imshow("Binary image" , image) ;
        vector<vector<Point> > contours ;
        findContours ( image , contours ,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE) ;
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
    */
    drawContours( drawing , contours , largestcontour , color , 1 , 8) ;
    namedWindow( "Contour" , CV_WINDOW_AUTOSIZE ) ;
    imshow( "Contour" , drawing ) ;
    waitKey (0) ;
}
