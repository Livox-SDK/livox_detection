#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <strings.h>
#include <assert.h>

#include <dirent.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fstream>
#include <sstream> 
BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)

typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > Polygon;

namespace py = pybind11;
using namespace std;

struct box{
  float ry;
  float l;
  float w;
  float h;
  float x;
  float z;
  float y;
  int cls_num;
  float is_obj;
  float wx0;
  float wx1;
  float wx2;
  float wx3;
  float wy0;
  float wy1;
  float wy2;
  float wy3;
  
};

void compute_4_points(vector<box> &boxes)
{
  for(int i=0;i<boxes.size();i++)
  {
    float ry = boxes[i].ry;
    float x = boxes[i].x; 
    float z = boxes[i].z;
    float l = boxes[i].l; 
    float w = boxes[i].w; 
    using namespace boost::numeric::ublas;
    using namespace boost::geometry;
    matrix<double> mref(2, 2);
    mref(0, 0) = cos(ry); mref(0, 1) = sin(ry);
    mref(1, 0) = -sin(ry); mref(1, 1) = cos(ry);

    matrix<double> corners(2, 4);
    double data[] = {l / 2, l / 2, -l / 2, -l / 2,
                     w / 2, -w / 2, -w / 2, w / 2};
    std::copy(data, data + 8, corners.data().begin());
    matrix<double> gc = prod(mref, corners);
    for (int i = 0; i < 4; ++i) {
        gc(0, i) += x;
        gc(1, i) += z;
    }
    boxes[i].wx0 = gc(0, 0); boxes[i].wy0=gc(1, 0);
    boxes[i].wx1 = gc(0, 1); boxes[i].wy1=gc(1, 1);
    boxes[i].wx2 = gc(0, 2); boxes[i].wy2=gc(1, 2);
    boxes[i].wx3 = gc(0, 3); boxes[i].wy3=gc(1, 3);
  }

}

Polygon toPolygon_py(double ry,double l,double w,double x,double z) 
{
    using namespace boost::numeric::ublas;
    using namespace boost::geometry;
    matrix<double> mref(2, 2);
    mref(0, 0) = cos(ry); mref(0, 1) = sin(ry);
    mref(1, 0) = -sin(ry); mref(1, 1) = cos(ry);

    static int count = 0;
    matrix<double> corners(2, 4);
    double data[] = {l / 2, l / 2, -l / 2, -l / 2,
                     w / 2, -w / 2, -w / 2, w / 2};
    std::copy(data, data + 8, corners.data().begin());
    matrix<double> gc = prod(mref, corners);
    for (int i = 0; i < 4; ++i) {
        gc(0, i) += x;
        gc(1, i) += z;
    }

    double points[][2] = {{gc(0, 0), gc(1, 0)},{gc(0, 1), gc(1, 1)},{gc(0, 2), gc(1, 2)},{gc(0, 3), gc(1, 3)},{gc(0, 0), gc(1, 0)}};
    Polygon poly;
    append(poly, points);
    return poly;
}
double groundBoxOverlap_py(double ry1,double l1,double w1,double x1,double z1,\
  double ry2,double l2,double w2,double x2,double z2, int criterion = -1)
{
  using namespace boost::geometry;
  Polygon gp = toPolygon_py(ry1,l1,w1,x1,z1);
  Polygon dp = toPolygon_py(ry2,l2,w2,x2,z2);

  std::vector<Polygon> in, un;
  intersection(gp, dp, in);
  union_(gp, dp, un);

  double inter_area = in.empty() ? 0 : area(in.front());
  double union_area = area(un.front());
  double o;
  if(criterion==-1)     // union
      o = inter_area / union_area;
  else if(criterion==0) // bbox_a
      o = inter_area / area(dp);
  else if(criterion==1) // bbox_b
      o = inter_area / area(gp);

  return o;
}


Polygon toPolygon(box a) 
{
    using namespace boost::numeric::ublas;
    using namespace boost::geometry;
    matrix<double> mref(2, 2);
    mref(0, 0) = cos(a.ry); mref(0, 1) = sin(a.ry);
    mref(1, 0) = -sin(a.ry); mref(1, 1) = cos(a.ry);

    static int count = 0;
    matrix<double> corners(2, 4);
    double data[] = {a.l / 2, a.l / 2, -a.l / 2, -a.l / 2,
                     a.w / 2, -a.w / 2, -a.w / 2, a.w / 2};
    std::copy(data, data + 8, corners.data().begin());
    matrix<double> gc = prod(mref, corners);
    for (int i = 0; i < 4; ++i) {
        gc(0, i) += a.x;
        gc(1, i) += a.z;
    }

    double points[][2] = {{gc(0, 0), gc(1, 0)},{gc(0, 1), gc(1, 1)},{gc(0, 2), gc(1, 2)},{gc(0, 3), gc(1, 3)},{gc(0, 0), gc(1, 0)}};
    Polygon poly;
    append(poly, points);
    return poly;
}


float compute_iou_ground(box a,box b, int criterion = -1)
{
  using namespace boost::geometry;
  Polygon gp = toPolygon(a);
  Polygon dp = toPolygon(b);

  std::vector<Polygon> in, un;
  intersection(gp, dp, in);
  union_(gp, dp, un);

  double inter_area = in.empty() ? 0 : area(in.front());
  double union_area = area(un.front());
  double o;
  if(criterion==-1)     // union
      o = inter_area / union_area;
  else if(criterion==0) // bbox_a
      o = inter_area / area(dp);
  else if(criterion==1) // bbox_b
      o = inter_area / area(gp);

  return o;
}


float compute_iou_rect(box rectA,box rectB)
{
  float xa1 = max(max(max(rectA.wx0,rectA.wx1),rectA.wx2),rectA.wx3);
  float xa0 = min(min(min(rectA.wx0,rectA.wx1),rectA.wx2),rectA.wx3);
  float ya1 = max(max(max(rectA.wy0,rectA.wy1),rectA.wy2),rectA.wy3);
  float ya0 = min(min(min(rectA.wy0,rectA.wy1),rectA.wy2),rectA.wy3);

  float xb1 = max(max(max(rectB.wx0,rectB.wx1),rectB.wx2),rectB.wx3);
  float xb0 = min(min(min(rectB.wx0,rectB.wx1),rectB.wx2),rectB.wx3);
  float yb1 = max(max(max(rectB.wy0,rectB.wy1),rectB.wy2),rectB.wy3);
  float yb0 = min(min(min(rectB.wy0,rectB.wy1),rectB.wy2),rectB.wy3);

  if (xa0 > xb1) { return 0.; }
	if (ya0 > yb1) { return 0.; }
	if ((xa1) < xb0) { return 0.; }
	if ((ya1) < yb0) { return 0.; }
	float colInt = min(xa1, xb1) - max(xa0, xb0);
	float rowInt = min(ya1, yb1) - max(ya0, yb0);
	float intersection = colInt * rowInt;
	float areaA = (xa1-xa0) * (ya1-ya0);
	float areaB = (xb1-xb0) * (yb1-yb0);
	float intersectionPercent = intersection / (areaA + areaB - intersection);
	return intersectionPercent;
}


float sigmoid(float x)
{
  float s = 1.0 / (1.0 + exp(-x));
  return s;
}

void nms2(
        const std::vector<box>& srcRects,
        const std::vector<float>& scores,
        std::vector<box>& resRects,
        float thresh,
        int neighbors = 0,
        float minScoresSum = 0.f
        )
{
    resRects.clear();

    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }

    assert(srcRects.size() == scores.size());

    std::multimap<float, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(std::pair<float, size_t>(scores[i], i));
    }

    while (idxs.size() > 0)
    {
        auto lastElem = --std::end(idxs);
        box rect1 = srcRects[lastElem->second];

        int neigborsCount = 0;
        float score = lastElem->first;
        float scoresSum = lastElem->first;

        idxs.erase(lastElem);

        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        { 
            box rect2 = srcRects[pos->second];
            float distance2 = (rect1.x-rect2.x)*(rect1.x-rect2.x)\
              +(rect1.z-rect2.z)*(rect1.z-rect2.z);
            if(distance2>15*15)
            {
              ++pos;
              continue;
            }
            
            float overlap=0;
            if((abs(rect1.ry)<70.0*3.14158/180 && abs(rect1.ry)>20.0*3.14158/180) || \
              (abs(rect2.ry)<70.0*3.14158/180 && abs(rect2.ry)>20.0*3.14158/180))
            {
              overlap  = compute_iou_ground(rect1,rect2);
            }
            {
              overlap  = compute_iou_rect(rect1,rect2);
            }
            if (overlap > thresh)
            {
                scoresSum += pos->first;
                pos = idxs.erase(pos);
                ++neigborsCount;
            }
            else
            {
                ++pos;
            }
        }
        if (neigborsCount >= neighbors &&
                scoresSum >= minScoresSum)
        {
            resRects.push_back(rect1);
        }
    }
}

py::array_t<float> cal_result(py::array_t<float> &feature_out,\
  float obj_th,float OVERLAP,float Z_MIN, int img_height,int img_width,float DX,float DY,float DZ,\
  float nms_th)
{

  auto feature_map = feature_out.unchecked<3>();

  int feature_height = img_height/8+0.5;
  int feature_width = img_width/8+0.5;
  int grid_height = img_height/feature_height+0.5;
  int grid_width = img_width/feature_width+0.5;

  std::vector<box> objs;
  std::vector<float> scores;
  objs.clear();
  scores.clear();

  float cut_dis = OVERLAP-Z_MIN;

  for(int height_i=0;height_i<feature_height;height_i++)
  {
    for(int width_i=0;width_i<feature_width;width_i++)
    {
      float is_obj = sigmoid(feature_map(height_i,width_i,0));
      
      float reg_dy = feature_map(height_i,width_i,13);
      reg_dy = reg_dy*grid_height;
      float center_y = height_i*grid_height+reg_dy;
      float m_y = (center_y*DY);
      float reg_dx = feature_map(height_i,width_i,11);
      reg_dx = reg_dx*grid_width;
      float center_x = width_i*grid_width+reg_dx;
      float m_x = (center_x-img_width/2)*DX;
      float sin_theta = feature_map(height_i,width_i,7);
      float cos_theta = feature_map(height_i,width_i,9);
      float theta = atan2(sin_theta,cos_theta)/2;
      float reg_ln_l = feature_map(height_i,width_i,17);
      float reg_l = exp(reg_ln_l);
      float reg_ln_h = feature_map(height_i,width_i,21);
      float reg_h = exp(reg_ln_h);
      if(m_y>cut_dis)
      {
        m_y=m_y-cut_dis-OVERLAP;
        if(m_y-reg_l/2<-10)
          is_obj*=0.2;
      }
      else
      {
        m_x*=-1;
        m_y=-1*(m_y-OVERLAP);
        if(m_y+reg_l/2>10)
          is_obj*=0.2;
      }

      if(m_y>100 && abs(theta)<45*3.14158/180)
      {
        is_obj*=0.2;
      }

      float m_obj_th = obj_th;
      if(m_y>100)
        m_obj_th=obj_th*0.5;
      if(m_y>150)
        m_obj_th=obj_th*0.4;
      if(m_y>180)
        m_obj_th=obj_th*0.3;

      if(is_obj>m_obj_th)
      {
        int cls_num=0;
        float is_cls0=feature_map(height_i,width_i,2);
        float is_cls1=feature_map(height_i,width_i,3);
        float is_cls2=feature_map(height_i,width_i,4);
        float reg_ln_w = feature_map(height_i,width_i,15);
        float reg_w = exp(reg_ln_w);
        float m_z = feature_map(height_i,width_i,19);   
        if(is_cls0>is_cls1 && is_cls0>is_cls2)
        {
          cls_num = 0;
        }
        else if(is_cls1>is_cls0 && is_cls1>is_cls2)
        {
          cls_num = 1;
          if (is_obj<0.88 && m_y<100 && abs(m_x)<40 && abs(m_x)>5 && m_y>10)
            continue;
        }
        else
        {
          cls_num = 2;
          if (is_obj<0.88 && m_y<100 && abs(m_x)<40 && abs(m_x)>5 && m_y>10)
            continue;
        }
        box one_obj;
        one_obj.ry = theta;
        one_obj.l = reg_l;
        one_obj.w = reg_w;
        one_obj.x = m_x;
        one_obj.z = m_y;
        one_obj.cls_num=cls_num;
        one_obj.is_obj=is_obj;
        one_obj.h = reg_h;
        one_obj.y = m_z;
        objs.push_back(one_obj);

        scores.push_back(is_obj);

      }

      is_obj= sigmoid(feature_map(height_i,width_i,1));
      if(is_obj>obj_th)
      {
        int cls_num=0;
        float is_cls3=feature_map(height_i,width_i,5);
        float is_cls4=feature_map(height_i,width_i,6);
        if(is_cls3>is_cls4)
        {
          cls_num = 3;
        }
        else
        {
          cls_num = 4;
        }
        
        float sin_theta = feature_map(height_i,width_i,8);
        float cos_theta = feature_map(height_i,width_i,10);
        float reg_dx = feature_map(height_i,width_i,12);
        float reg_dy = feature_map(height_i,width_i,14);
        float reg_ln_w = feature_map(height_i,width_i,16);
        float reg_ln_l = feature_map(height_i,width_i,18);
        float m_z = feature_map(height_i,width_i,20);
        float reg_ln_h = feature_map(height_i,width_i,22);
        float theta = atan2(sin_theta,cos_theta)/2;
        reg_dx = reg_dx*grid_width;
        reg_dy = reg_dy*grid_height;
        float center_x = width_i*grid_width+reg_dx;
        float center_y = height_i*grid_height+reg_dy;
        float m_x = (center_x-img_width/2)*DX;
        float m_y = (center_y*DY);
        float reg_w = exp(reg_ln_w);
        float reg_l = exp(reg_ln_l);
        float reg_h = exp(reg_ln_h);

        if(m_y>cut_dis)
        {
          m_y=m_y-cut_dis-OVERLAP;
        }
        else
        {
          m_x*=-1;
          m_y=-1*(m_y-OVERLAP);
        }
   

        box one_obj;
        one_obj.ry = theta;
        one_obj.l = reg_l;
        one_obj.w = reg_w;
        one_obj.x = m_x;
        one_obj.z = m_y;
        one_obj.cls_num=cls_num;
        one_obj.is_obj=is_obj;
        one_obj.h = reg_h;
        one_obj.y = m_z;
        objs.push_back(one_obj);

        scores.push_back(is_obj);

      }
    }
  }

  std::vector<box> results;
  compute_4_points(objs);
  nms2(objs,scores,results,nms_th);

  int obj_num = results.size();

  auto result = py::array_t<float>(obj_num*9);
  result.resize({obj_num,9});
  py::buffer_info buf_result = result.request();
  float* ptr_result = (float*)buf_result.ptr;

  for(int i=0;i<obj_num;i++)
  {
    ptr_result[i*9 + 0] = results[i].is_obj;
    ptr_result[i*9 + 1] = results[i].cls_num;
    ptr_result[i*9 + 2] = results[i].ry;
    ptr_result[i*9 + 3] = results[i].l;
    ptr_result[i*9 + 4] = results[i].w;
    ptr_result[i*9 + 5] = results[i].x;
    ptr_result[i*9 + 6] = results[i].z;
    ptr_result[i*9 + 7] = results[i].h;
    ptr_result[i*9 + 8] = results[i].y;
  }

  return result;
}

PYBIND11_MODULE(lib_cpp, m) 
{
    m.def("cal_result", &cal_result);
}




int32_t main() {


  return 0;
}

