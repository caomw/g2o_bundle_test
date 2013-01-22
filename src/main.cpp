// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <Eigen/StdVector>
#include <iostream>
#include <stdint.h>

#ifdef _MSC_VER
#include <unordered_set>
#else
#include <tr1/unordered_set>
#endif

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
//#include "g2o/math_groups/se3quat.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#include "GraphBundler.hpp"

using namespace Eigen;
using namespace std;


class Sample {
public:
  static int uniform(int from, int to);
  static double uniform();
  static double gaussian(double sigma);
};

static double uniform_rand(double lowerBndr, double upperBndr){
  return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
}

static double gauss_rand(double mean, double sigma){
  double x, y, r2;
  do {
    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
    r2 = x * x + y * y;
  } while (r2 > 1.0 || r2 == 0.0);
  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
}

int Sample::uniform(int from, int to){
  return static_cast<int>(uniform_rand(from, to));
}

double Sample::uniform(){
  return uniform_rand(0., 1.);
}

double Sample::gaussian(double sigma){
  return gauss_rand(0., sigma);
}

int main(int argc, const char* argv[]){
  if (argc<2)
  {
    cout << endl;
    cout << "Please type: " << endl;
    cout << "ba_demo [PIXEL_NOISE] [OUTLIER RATIO] [ROBUST_KERNEL] [STRUCTURE_ONLY] [DENSE]" << endl;
    cout << endl;
    cout << "PIXEL_NOISE: noise in image space (E.g.: 1)" << endl;
    cout << "OUTLIER_RATIO: probability of spuroius observation  (default: 0.0)" << endl;
    cout << "ROBUST_KERNEL: use robust kernel (0 or 1; default: 0==false)" << endl;
    cout << "STRUCTURE_ONLY: performe structure-only BA to get better point initializations (0 or 1; default: 0==false)" << endl;
    cout << "DENSE: Use dense solver (0 or 1; default: 0==false)" << endl;
    cout << endl;
    cout << "Note, if OUTLIER_RATIO is above 0, ROBUST_KERNEL should be set to 1==true." << endl;
    cout << endl;
    exit(0);
  }

  double PIXEL_NOISE = atof(argv[1]);
  double OUTLIER_RATIO = 0.0;

  if (argc>2)  {
    OUTLIER_RATIO = atof(argv[2]);
  }

  bool ROBUST_KERNEL = false;
  if (argc>3){
    ROBUST_KERNEL = atoi(argv[3]) != 0;
  }
  bool STRUCTURE_ONLY = false;
  if (argc>4){
    STRUCTURE_ONLY = atoi(argv[4]) != 0;
  }

  bool DENSE = false;
  if (argc>5){
    DENSE = atoi(argv[5]) != 0;
  }

  cout << "PIXEL_NOISE: " <<  PIXEL_NOISE << endl;
  cout << "OUTLIER_RATIO: " << OUTLIER_RATIO<<  endl;
  cout << "ROBUST_KERNEL: " << ROBUST_KERNEL << endl;
  cout << "STRUCTURE_ONLY: " << STRUCTURE_ONLY<< endl;
  cout << "DENSE: "<<  DENSE << endl;

	GraphBundler bundler(DENSE,ROBUST_KERNEL);
	SyntheticWorldGenerator world(500,1000.0,Vector2d(320.,240.));

	if (!bundler.addCameraParams(world.getFocalLength(),world.getPrincipalPoint()))
	{
		assert(false);
	}

 	for (size_t i=0; i<world.numPoses(); ++i) 
	{
		bundler.addPoseVertex(world.getPose(i));
	}

	int point_id=0;
	int point_num = 0;
	double sum_diff2 = 0;

	tr1::unordered_map<int,int> pointid_2_trueid;
	tr1::unordered_set<int> inliers;

	//! Add points and observations.
	for (size_t i=0; i< world.numPoints(); ++i)
	{
		point_id = bundler.addPointVertex(world.getPointWithNoise(i));
	
		int num_obs = 0;
		
		//! Count number of visible points
		for (size_t j=0; j<world.numPoses(); ++j)
		{
			Vector2d z = bundler.predict(world.getPose(j).map(world.getPoint(i)));
			if(world.isInImage(z))
			{
				++num_obs;
			}
		}

		if (num_obs>=2)
		{
			bool inlier = true;
			
			for (size_t j=0; j<world.numPoses(); ++j)
			{
				Vector2d z = bundler.predict(world.getPose(j).map(world.getPoint(i)));

				if (world.isInImage(z))
				{
					double sam = Sample::uniform();
					
					//! Simulates outlier situation
					if (sam<OUTLIER_RATIO)
					{
						z = Vector2d(Sample::uniform(0,640),Sample::uniform(0,480));
						inlier= false;
					}
					
					z += Vector2d(Sample::gaussian(PIXEL_NOISE),Sample::gaussian(PIXEL_NOISE));
					bundler.setConstraint(i,j,z);
				}
			}/// end for

			if (inlier)
			{
				inliers.insert(point_id);
				Vector3d diff = bundler.getPointFromGraph(point_id) - world.getPoint(i);
				sum_diff2 += diff.dot(diff);
			} /// end if
			pointid_2_trueid.insert(make_pair(point_id,i));
			++point_num;
		}/// end if (num_obs>=2)
	}/// end for (size_t i=0; i< world.numPoints(); ++i)

	cout << endl;

	cout << "Performing full BA:" << endl;
	bundler.doBundleAdjustment(15);
	cout << "Done" << endl;
	
	cout << "Point error before optimisation (inliers only): " << sqrt(sum_diff2/point_num) << endl;
	
	sum_diff2 = 0;
	point_num = 0;
	
	for(size_t i=0;i<bundler.numPoint();++i)
	{
		Vector3d diff = bundler.getPointFromResult(i) - world.getPoint(i);
		sum_diff2 += diff.dot(diff);
		++point_num;
	}
	
	cout << "Point error after optimisation (inliers only): " << sqrt(sum_diff2/point_num) << endl;
	
}

