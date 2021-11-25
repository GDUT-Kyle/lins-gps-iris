#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

//----------------------------------------------------------
// Recombinate image quaters
//----------------------------------------------------------
void Recomb(Mat &src, Mat &dst)
{
    int cx = src.cols >> 1;
    int cy = src.rows >> 1;
    Mat tmp;
    tmp.create(src.size(), src.type());
    src(Rect(0, 0, cx, cy)).copyTo(tmp(Rect(cx, cy, cx, cy)));
    src(Rect(cx, cy, cx, cy)).copyTo(tmp(Rect(0, 0, cx, cy)));
    src(Rect(cx, 0, cx, cy)).copyTo(tmp(Rect(0, cy, cx, cy)));
    src(Rect(0, cy, cx, cy)).copyTo(tmp(Rect(cx, 0, cx, cy)));
    dst = tmp;
}
//----------------------------------------------------------
// 2D Forward FFT
//----------------------------------------------------------
void ForwardFFT(Mat &Src, Mat *FImg, bool do_recomb = true)
{
    int M = getOptimalDFTSize(Src.rows);
    int N = getOptimalDFTSize(Src.cols);
    Mat padded;
    copyMakeBorder(Src, padded, 0, M - Src.rows, 0, N - Src.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
    Mat complexImg;
    merge(planes, 2, complexImg);
    dft(complexImg, complexImg);
    split(complexImg, planes);
    planes[0] = planes[0](Rect(0, 0, planes[0].cols & -2, planes[0].rows & -2));
    planes[1] = planes[1](Rect(0, 0, planes[1].cols & -2, planes[1].rows & -2));
    if (do_recomb)
    {
        Recomb(planes[0], planes[0]);
        Recomb(planes[1], planes[1]);
    }
    planes[0] /= float(M*N);
    planes[1] /= float(M*N);
    FImg[0] = planes[0].clone();
    FImg[1] = planes[1].clone();
}
//----------------------------------------------------------
// 2D inverse FFT
//----------------------------------------------------------
void InverseFFT(Mat *FImg, Mat &Dst, bool do_recomb = true)
{
    if (do_recomb)
    {
        Recomb(FImg[0], FImg[0]);
        Recomb(FImg[1], FImg[1]);
    }
    Mat complexImg;
    merge(FImg, 2, complexImg);
    idft(complexImg, complexImg);
    split(complexImg, FImg);
    Dst = FImg[0].clone();
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
void highpass(Size sz, Mat& dst)
{
    Mat a = Mat(sz.height, 1, CV_32FC1);
    Mat b = Mat(1, sz.width, CV_32FC1);

    float step_y = CV_PI / sz.height;
    float val = -CV_PI*0.5;

    for (int i = 0; i < sz.height; ++i)
    {
        a.at<float>(i) = cos(val);
        val += step_y;
    }

    val = -CV_PI*0.5;
    float step_x = CV_PI / sz.width;
    for (int i = 0; i < sz.width; ++i)
    {
        b.at<float>(i) = cos(val);
        val += step_x;
    }

    Mat tmp = a*b;
    dst = (1.0 - tmp).mul(2.0 - tmp);
}
//-----------------------------------------------------------------------------------------------------
//
//-----------------------------------------------------------------------------------------------------
float logpolar(Mat& src, Mat& dst)
{
    float radii = src.cols;
    float angles = src.rows;
    Point2f center(src.cols / 2, src.rows / 2);
    float d = norm(Vec2f(src.cols - center.x, src.rows - center.y));
    float log_base = pow(10.0, log10(d) / radii);
    float d_theta = CV_PI / (float)angles;
    float theta = CV_PI / 2.0;
    float radius = 0;
    Mat map_x(src.size(), CV_32FC1);
    Mat map_y(src.size(), CV_32FC1);
    for (int i = 0; i < angles; ++i)
    {
        for (int j = 0; j < radii; ++j)
        {
            radius = pow(log_base, float(j));
            float x = radius * sin(theta) + center.x;
            float y = radius * cos(theta) + center.y;
            map_x.at<float>(i, j) = x;
            map_y.at<float>(i, j) = y;
        }
        theta += d_theta;
    }
    remap(src, dst, map_x, map_y, cv::INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    return log_base;
}
//-----------------------------------------------------------------------------------------------------
// As input we need equal sized images, with the same aspect ratio,
// scale difference should not exceed 1.8 times.
//-----------------------------------------------------------------------------------------------------
RotatedRect LogPolarFFTTemplateMatch(Mat& im0, Mat& im1/*, double canny_threshold1, double canny_threshold2*/)
{
    // Accept 1 or 3 channel CV_8U, CV_32F or CV_64F images.
    CV_Assert((im0.type() == CV_8UC1) || (im0.type() == CV_8UC3) ||
        (im0.type() == CV_32FC1) || (im0.type() == CV_32FC3) ||
        (im0.type() == CV_64FC1) || (im0.type() == CV_64FC3));

    CV_Assert(im0.rows == im1.rows && im0.cols == im1.cols);

    CV_Assert(im0.channels() == 1 || im0.channels() == 3 || im0.channels() == 4);

    CV_Assert(im1.channels() == 1 || im1.channels() == 3 || im1.channels() == 4);

    //Mat im0_tmp = im0.clone();
    //Mat im1_tmp = im1.clone();
    if (im0.channels() == 3)
    {
        cvtColor(im0, im0, cv::COLOR_BGR2GRAY);
    }

    if (im0.channels() == 4)
    {
        cvtColor(im0, im0, cv::COLOR_BGRA2GRAY);
    }

    if (im1.channels() == 3)
    {
        cvtColor(im1, im1, cv::COLOR_BGR2GRAY);
    }

    if (im1.channels() == 4)
    {
        cvtColor(im1, im1, cv::COLOR_BGRA2GRAY);
    }

    if (im0.type() == CV_32FC1)
    {
       im0.convertTo(im0, CV_8UC1, 255.0);
    }

    if (im1.type() == CV_32FC1)
    {
       im1.convertTo(im1, CV_8UC1, 255.0);
    }

    if (im0.type() == CV_64FC1)
    {
        im0.convertTo(im0, CV_8UC1, 255.0);
    }

    if (im1.type() == CV_64FC1)
    {
        im1.convertTo(im1, CV_8UC1, 255.0);
    }


    // Canny(im0, im0, canny_threshold1, canny_threshold2); // you can change this
    // Canny(im1, im1, canny_threshold1, canny_threshold2);
    
    // Ensure both images are of CV_32FC1 type
    im0.convertTo(im0, CV_32FC1, 1.0 / 255.0);
    im1.convertTo(im1, CV_32FC1, 1.0 / 255.0);

    Mat F0[2], F1[2];
    Mat f0, f1;
    // 前向快速傅里叶变换
    ForwardFFT(im0, F0);
    ForwardFFT(im1, F1);
    // 该函数用来计算虚数的幅值 
    magnitude(F0[0], F0[1], f0);
    magnitude(F1[0], F1[1], f1);

    // Create filter 
    // 高通滤波器?
    Mat h;
    highpass(f0.size(), h);

    // Apply it in freq domain
    // 在频域上进行高通滤波
    f0 = f0.mul(h);
    f1 = f1.mul(h);

    float log_base;
    Mat f0lp, f1lp;

    // 将图像映射到极坐标
    // 参考:https://blog.csdn.net/BeBuBu/article/details/102517138
    // 图像的尺寸的差异被旋换为对数极坐标平面内沿着p轴的位移，旋转差异被转换成对数极坐标平面沿θ轴的位移。
    log_base = logpolar(f0, f0lp);
    log_base = logpolar(f1, f1lp);

    // Find rotation and scale
    // 相位相关法（phase correclate）:用于检测两幅内容相同的图像之间的相对位移量(注意:这里的图像是LogPolar形式,
    // 因此是寻找旋转和尺度的关系)
    Point2d rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);

    float angle = 180.0 * rotation_and_scale.y / f0lp.rows;
    float scale = pow(log_base, rotation_and_scale.x);
    // --------------
    if (scale > 1.8)
    {
        rotation_and_scale = cv::phaseCorrelate(f1lp, f0lp);
        angle = -180.0 * rotation_and_scale.y / f0lp.rows;
        scale = 1.0 / pow(log_base, rotation_and_scale.x);
        if (scale > 1.8)
        {
            cout << "Images are not compatible. Scale change > 1.8" << endl;
            return RotatedRect();
        }
    }
    // --------------
    if (angle < -90.0)
    {
        angle += 180.0;
    }
    else if (angle > 90.0)
    {
        angle -= 180.0;
    }

    // Now rotate and scale fragment back, then find translation
    // 现在旋转并缩放片段，然后找到偏移
    // Mat getRotationMatrix2D(Point2f center, double angle, double scale) 生成旋转矩阵
    // Point2f center：表示旋转的中心点
    // double angle：表示旋转的角度
    // double scale：图像缩放因子
    Mat rot_mat = getRotationMatrix2D(Point(im1.cols / 2, im1.rows / 2), angle, 1.0 / scale);

    // rotate and scale
    Mat im1_rs;
    // 对图像进行仿射变换,使得im1_rs和im0的尺度和旋转相同
    warpAffine(im1, im1_rs, rot_mat, im1.size());

    // find translation
    // 相位相关法（phase correlate）可以用于检测两幅内容相同的图像之间的相对位移量。
    // 它是基于傅立叶变换的位移定理：一个平移过的函数的傅立叶变换仅仅是未平移函数的傅立
    // 叶变换与一个具有线性相位的指数因子的乘积，即空间域中的平移会造成频域中频谱的相移。
    // 计算两图像间的偏移(注意:这里是原图像,不是LogPolar)
    Point2d tr = cv::phaseCorrelate(im1_rs, im0);

    // compute rotated rectangle parameters
    RotatedRect rr;
    rr.center = tr + Point2d(im0.cols / 2, im0.rows / 2);
    rr.angle = -angle;
    rr.size.width = im1.cols / scale;
    rr.size.height = im1.rows / scale;

    //im0 = im0_tmp.clone();
    //im1 = im1_tmp.clone();

    return rr;
}

RotatedRect FFTMatch(const Mat& im0, const Mat& im1)
{
    Mat im0_tmp = im0.clone();
    Mat im1_tmp = im1.clone();
    // 基于FFT的模板匹配
    return LogPolarFFTTemplateMatch(im0_tmp, im1_tmp);
}
