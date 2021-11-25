#include "LidarIris.h"
#include "fftm/fftm.hpp"

using namespace Eigen;

cv::Mat1b LidarIris::GetIris(const pcl::PointCloud<pcl::PointXYZI> &cloud)
{
    // LiDAR-Iris Image Representation
    // 对点云投影进行栅格分割后保存于的矩阵
    cv::Mat1b IrisMap = cv::Mat1b::zeros(80, 360);

    // 16-line
    for (pcl::PointXYZI p : cloud.points)
    {
        float dis = sqrt(p.data[0] * p.data[0] + p.data[1] * p.data[1]);
        float arc = (atan2(p.data[2], dis) * 180.0f / M_PI) + 15;
        float yaw = (atan2(p.data[1], p.data[0]) * 180.0f / M_PI) + 180;
        int Q_dis = std::min(std::max((int)floor(dis), 0), 79);
        int Q_arc = std::min(std::max((int)floor(arc / 4.0f), 0), 7);
        int Q_yaw = std::min(std::max((int)floor(yaw + 0.5), 0), 359);
        IrisMap.at<uint8_t>(Q_dis, Q_yaw) |= (1 << Q_arc);
    }


    // 64-line
    // 遍历点云的点
    // for (pcl::PointXYZI p : cloud.points)
    // {
    //     // 点离雷达底座的距离
    //     float dis = sqrt(p.data[0] * p.data[0] + p.data[1] * p.data[1]);
    //     float arc = (atan2(p.data[2], dis) * 180.0f / M_PI) + 24;
    //     float yaw = (atan2(p.data[1], p.data[0]) * 180.0f / M_PI) + 180;
    //     int Q_dis = std::min(std::max((int)floor(dis), 0), 79);
    //     int Q_arc = std::min(std::max((int)floor(arc / 4.0f), 0), 7);
    //     int Q_yaw = std::min(std::max((int)floor(yaw + 0.5), 0), 359);
    //     IrisMap.at<uint8_t>(Q_dis, Q_yaw) |= (1 << Q_arc);
    // }
    return IrisMap;
}

void LidarIris::UpdateFrame(const cv::Mat1b &frame, int frameIndex, float *matchDistance, int *matchIndex)
{
    // first: calc feature
    std::vector<float> vec;
    auto feature = GetFeature(frame, vec);
    flann::Matrix<float> queries(vec.data(), 1, vec.size());
    if (featureList.size() == 0)
    {
        if (matchDistance)
            *matchDistance = NAN;
        if (matchIndex)
            *matchIndex = -1;
        vecList.buildIndex(queries);
    }
    else
    {
        // second: search in database
        vecList.knnSearch(queries, indices, dists, _matchNum, flann::SearchParams(32));
        //thrid: calc matches
        std::vector<float> dis(_matchNum);
        for (int i = 0; i < _matchNum; i++)
        {
            dis[i] = Compare(feature, featureList[indices[0][i]]);
        }
        int minIndex = std::min_element(dis.begin(), dis.end()) - dis.begin();
        if (matchDistance)
            *matchDistance = dis[minIndex];
        if (matchIndex)
            *matchIndex = frameIndexList[indices[0][minIndex]];
        // forth: add frame to database
        vecList.addPoints(queries);
    }
    featureList.push_back(feature);
    frameIndexList.push_back(frameIndex);
}

float LidarIris::Compare(const LidarIris::FeatureDesc &img1, const LidarIris::FeatureDesc &img2, int *bias)
{
    // 快速傅里叶变换模板匹配找到图像间的偏移
    auto firstRect = FFTMatch(img2.img, img1.img);
    int firstShift = firstRect.center.x - img1.img.cols / 2;
    float dis1;
    int bias1;
    // 根据匹配的偏移量变换后计算汉明距离
    GetHammingDistance(img1.T, img1.M, img2.T, img2.M, firstShift, dis1, bias1);
    //
    auto T2x = circShift(img2.T, 0, 180);
    auto M2x = circShift(img2.M, 0, 180);
    auto img2x = circShift(img2.img, 0, 180);
    //
    auto secondRect = FFTMatch(img2x, img1.img);
    int secondShift = secondRect.center.x - img1.img.cols / 2;
    float dis2 = 0;
    int bias2 = 0;
    GetHammingDistance(img1.T, img1.M, T2x, M2x, secondShift, dis2, bias2);
    //
    if (dis1 < dis2)
    {
        if (bias)
            *bias = bias1;
        return dis1;
    }
    else
    {
        if (bias)
            *bias = (bias2 + 180) % 360;
        return dis2;
    }
}

// Log-Gabor filter:https://xuewenyuan.github.io/posts/2016/05/blog-post-1/

// Parameters:
// nscale: number of wavelet scales(4)
// minWaveLength: wavelength of smallest scale filter(18)
// mult: scaling factor between successive filters.(1.6)
// sigmaOnf: Ratio of the standard deviation of the Gaussian describing(0.75)
//           the log Gabor filter's transfer function in the frequency 
//           domain to the filter center frequency.
std::vector<cv::Mat2f> LidarIris::LogGaborFilter(const cv::Mat1f &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf)
{
    // iris的行数
    int rows = src.rows;
    // 列数
    int cols = src.cols;
    cv::Mat2f filtersum = cv::Mat2f::zeros(1, cols);
    // 存储不同尺度下的特征图
    std::vector<cv::Mat2f> EO(nscale);
    // logGabor内核长度偶数化
    int ndata = cols;
    if (ndata % 2 == 1)
        ndata--;
    // 滤波器内核
    cv::Mat1f logGabor = cv::Mat1f::zeros(1, ndata);
    cv::Mat2f result = cv::Mat2f::zeros(rows, ndata);
    cv::Mat1f radius = cv::Mat1f::zeros(1, ndata / 2 + 1);
    radius.at<float>(0, 0) = 1;
    for (int i = 1; i < ndata / 2 + 1; i++)
    {
        radius.at<float>(0, i) = i / (float)ndata;
    }
    // 波长
    double wavelength = minWaveLength;
    // 遍历所有尺度
    for (int s = 0; s < nscale; s++)
    {
        // Centre frequency of filter.
        // 滤波器中心频率
        double fo = 1.0 / wavelength;
        double rfo = fo / 0.5;
        // Log Gabor function(频域!!) 好像是预构建一个1D的卷积核
        cv::Mat1f temp; //(radius.size());
        cv::log(radius / fo, temp);
        cv::pow(temp, 2, temp);
        cv::exp((-temp) / (2 * log(sigmaOnf) * log(sigmaOnf)), temp);
        temp.copyTo(logGabor.colRange(0, ndata / 2 + 1));
        // Log Gabor function

        // Log Gabor filter
        logGabor.at<float>(0, 0) = 0;
        cv::Mat2f filter;
        cv::Mat1f filterArr[2] = {logGabor, cv::Mat1f::zeros(logGabor.size())};
        cv::merge(filterArr, 2, filter);
        filtersum = filtersum + filter;
        // 对iris的每一行进行log-gabor滤波器内核卷积?
        for (int r = 0; r < rows; r++)
        {
            cv::Mat2f src2f;
            // 以iris的行作为对象进行滤波操作
            cv::Mat1f srcArr[2] = {src.row(r).clone(), cv::Mat1f::zeros(1, src.cols)};
            cv::merge(srcArr, 2, src2f);
            // 《学习OpenCV（中文版）》-- P200
            // 这里需要了解卷积定理公式,实际操作是卷积,但是它先将其转到频域上,再进行频谱相乘,再转回时域
            // 离散傅里叶变换
            cv::dft(src2f, src2f);
            // MulSpectrums 是对于两张频谱图中每个元素的乘法。在频域上进行log-gabor滤波
            cv::mulSpectrums(src2f, filter, src2f, 0);
            // 离散傅里叶逆变换
            cv::idft(src2f, src2f);
            // 滤波后将结果合成到result
            src2f.copyTo(result.row(r));
        }
        // 完成一个尺度下的log-gabor滤波,将result放到EO中
        EO[s] = result.clone();
        // 改变滤波器波长,也就是中心频率fo
        wavelength *= mult;
    }
    filtersum = circShift(filtersum, 0, cols / 2);
    return EO;
}

void LidarIris::LoGFeatureEncode(const cv::Mat1b &src, unsigned int nscale, int minWaveLength, double mult, double sigmaOnf, cv::Mat1b &T, cv::Mat1b &M)
{
    cv::Mat1f srcFloat;
    src.convertTo(srcFloat, CV_32FC1);
    auto list = LogGaborFilter(srcFloat, nscale, minWaveLength, mult, sigmaOnf);
    std::vector<cv::Mat1b> Tlist(nscale * 2), Mlist(nscale * 2); // bool图像
    // list.size()==尺度数?
    for (int i = 0; i < list.size(); i++)
    {
        cv::Mat1f arr[2];
        cv::split(list[i], arr);
        // 记录实部和虚部,相当于相位?
        Tlist[i] = arr[0] > 0; // 实部 判断是否>0,若是,Tlist[i]=true, else Tlist[i]=false
        Tlist[i + nscale] = arr[1] > 0; // 虚部
        cv::Mat1f m;
        // 计算幅值
        cv::magnitude(arr[0], arr[1], m);
        Mlist[i] = m < 0.0001;
        Mlist[i + nscale] = m < 0.0001;
    }
    // 按行合并?
    cv::vconcat(Tlist, T);
    cv::vconcat(Mlist, M);
}

LidarIris::FeatureDesc LidarIris::GetFeature(const cv::Mat1b &src)
{
    // 创建特征描述子
    FeatureDesc desc;
    // 将iris image存于特征结构体
    desc.img = src;
    // 傅里叶变换-(转到频域)->log-gabor滤波器-(转回时域)->傅里叶逆变换
    LoGFeatureEncode(src, _nscale, _minWaveLength, _mult, _sigmaOnf, desc.T, desc.M);
    return desc;
}

LidarIris::FeatureDesc LidarIris::GetFeature(const cv::Mat1b &src, std::vector<float> &vec)
{
    cv::Mat1f temp;
    src.convertTo(temp, CV_32FC1);
    cv::reduce((temp != 0) / 255, temp, 1, cv::REDUCE_AVG);
    vec = temp.isContinuous() ? temp : temp.clone();
    return GetFeature(src);
}

void LidarIris::GetHammingDistance(const cv::Mat1b &T1, const cv::Mat1b &M1, const cv::Mat1b &T2, const cv::Mat1b &M2, int scale, float &dis, int &bias)
{
    dis = NAN;
    bias = -1;
    for (int shift = scale - 2; shift <= scale + 2; shift++)
    {
        cv::Mat1b T1s = circShift(T1, 0, shift);
        cv::Mat1b M1s = circShift(M1, 0, shift);
        cv::Mat1b mask = M1s | M2;
        int MaskBitsNum = cv::sum(mask / 255)[0];
        int totalBits = T1s.rows * T1s.cols - MaskBitsNum;
        cv::Mat1b C = T1s ^ T2;
        C = C & ~mask;
        int bitsDiff = cv::sum(C / 255)[0];
        if (totalBits == 0)
        {
            dis = NAN;
        }
        else
        {
            float currentDis = bitsDiff / (float)totalBits;
            if (currentDis < dis || std::isnan(dis))
            {
                dis = currentDis;
                bias = shift;
            }
        }
    }
    return;
}

inline cv::Mat LidarIris::circRowShift(const cv::Mat &src, int shift_m_rows)
{
    if (shift_m_rows == 0)
        return src.clone();
    shift_m_rows %= src.rows;
    int m = shift_m_rows > 0 ? shift_m_rows : src.rows + shift_m_rows;
    if(src.rows - m ==0)
        return src.clone();
    cv::Mat dst(src.size(), src.type());
    src(cv::Range(src.rows - m, src.rows), cv::Range::all()).copyTo(dst(cv::Range(0, m), cv::Range::all()));
    src(cv::Range(0, src.rows - m), cv::Range::all()).copyTo(dst(cv::Range(m, src.rows), cv::Range::all()));
    return dst;
}

inline cv::Mat LidarIris::circColShift(const cv::Mat &src, int shift_n_cols)
{
    if (shift_n_cols == 0)
        return src.clone();
    shift_n_cols %= src.cols;
    int n = shift_n_cols > 0 ? shift_n_cols : src.cols + shift_n_cols;
    if(src.cols - n ==0)
        return src.clone();
    cv::Mat dst(src.size(), src.type());
    // 1.Range是OpenCV中新加入的一个类，该类有两个关键的变量start和end；
    // 2.Range对象可以用来表示矩阵的多个连续的行或者多个连续的列
    // 3.Range表示范围从start到end，包含start，但不包含end；
    // 4.Range类还提供了一个静态方法all()，这个方法的作用如同Matlab中的“：”,表示所有的行或者所有的列
    // // 1.创建一个单位阵
    // Mat A= Mat::eye(10, 10, CV_32S);
    // // 2.提取第1到3列（不包括3）
    // Mat B = A(Range::all(),Range(1,3));
    // // 3.提取B的第5至9行（不包括9）
    // C= B(Range(5,9),Range::all());
    src(cv::Range::all(), cv::Range(src.cols - n, src.cols)).copyTo(dst(cv::Range::all(), cv::Range(0, n)));
    src(cv::Range::all(), cv::Range(0, src.cols - n)).copyTo(dst(cv::Range::all(), cv::Range(n, src.cols)));
    return dst;
}

cv::Mat LidarIris::circShift(const cv::Mat &src, int shift_m_rows, int shift_n_cols)
{
    // 循环行列变换
    return circColShift(circRowShift(src, shift_m_rows), shift_n_cols);
}