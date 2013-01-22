#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>

#include <tr1/unordered_map>

class SyntheticWorldGenerator
{
public:
	SyntheticWorldGenerator(unsigned int nPoints, double f, const Eigen::Vector2d c): focal_length(f), principal_point(c)
	{
		//! Make world.
		for(size_t i=0;i<nPoints;++i)
		{
			true_points.push_back(
					Eigen::Vector3d(
									(uniform()-0.5)*3,
		                            uniform()-0.5,
		                            uniform()+3)
									);
		}
		
		//! Make poses.
		for(size_t i=0;i<15;++i)
		{
			Eigen::Vector3d trans(i*0.04-1.0,0,0);
			Eigen:: Quaterniond q;
		    q.setIdentity();
		    g2o::SE3Quat pose(q,trans);
			true_poses.push_back(pose);
		}
	}
	
	~SyntheticWorldGenerator()
	{
		
	}
	
	bool isInImage(const Eigen::Vector2d &z)
	{
		return (z[0]>=0 && z[1]>=0 && z[0]<640 && z[1]<480);
	}
	
	double getFocalLength(void)
	{
		return focal_length;
	}
		
	Eigen::Vector2d getPrincipalPoint()
	{
		return principal_point;
	}
	
	int numPoses()
	{
		return true_poses.size();
	}
	
	int numPoints()
	{
		return true_points.size();
	}
	
	Eigen::Vector3d getPointWithNoise(int idx)
	{
		return true_points.at(idx) + Eigen::Vector3d(gaussian(1),gaussian(1),gaussian(1));
	}
	
	Eigen::Vector3d getPoint(int idx)
	{
		return true_points.at(idx);
	}
	
	g2o::SE3Quat getPose(int idx)
	{
		return true_poses.at(idx);
	}
	
private:
	std::vector<Eigen::Vector3d> true_points;
	std::vector<g2o::SE3Quat,Eigen::aligned_allocator<g2o::SE3Quat> > true_poses;
	
	double focal_length;
	Eigen::Vector2d principal_point;
	
	// Random related function
	int uniform(int from, int to){
	  return static_cast<int>(uniform_rand(from, to));
	}
	
	double uniform(){
	  return uniform_rand(0., 1.);
	}
	
	double gaussian(double sigma){
	  return gauss_rand(0., sigma);
	}
	
	double uniform_rand(double lowerBndr, double upperBndr){
	  return lowerBndr + ((double) std::rand() / (RAND_MAX + 1.0)) * (upperBndr - lowerBndr);
	}
	
	double gauss_rand(double mean, double sigma){
	  double x, y, r2;
	  do {
	    x = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
	    y = -1.0 + 2.0 * uniform_rand(0.0, 1.0);
	    r2 = x * x + y * y;
	  } while (r2 > 1.0 || r2 == 0.0);
	  return mean + sigma * y * std::sqrt(-2.0 * log(r2) / r2);
	}
};

class GraphBundler
{
public:
	//! Data structures	
	typedef std::vector<g2o::SE3Quat> PoseVector;
	typedef std::vector<Eigen::Vector3d> PointVector;
	
	GraphBundler(bool isDenseOptimizer, bool useRobustKernel) : unique_id(-1)
	{
		optimizer = new g2o::SparseOptimizer;
		optimizer->setVerbose(false);
		
		if(isDenseOptimizer)
		{
			linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
		}
		else
		{
			linearSolver = new g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType>();
		}
		
		solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
		solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
		
		optimizer->setAlgorithm(solver);
		
		robustKernel = useRobustKernel;
	};
	
	~GraphBundler(){}
	
	void doBundleAdjustment(int numIterations = 10)
	{
		optimizer->initializeOptimization();
		optimizer->setVerbose(true);
		optimizer->optimize(numIterations);
		
		//! result back to origin
		//! for each pose vector
		for(size_t i=0;i<vPose.size();++i)
		{
			int vid = PoseVectorId_to_VertexId[i];
			//! fetch graph vertex
			g2o::HyperGraph::VertexIDMap::iterator v_it = optimizer->vertices().find(vid);
			//! check sanity
			if(v_it == optimizer->vertices().end())
			{
				std::cerr << "Vertex " << vid << " not in graph!" << std::endl;
				exit(-1);
			}
			g2o::VertexSE3Expmap * v_se3 = dynamic_cast< g2o::VertexSE3Expmap * > (v_it->second);
			vPose[i] = v_se3->estimate();	//! return back
		}
		
		//! for each point vector
		for(size_t i=0;i<vPoint.size();++i)
		{
			int vid = PointVectorId_to_VertexId[i];
			g2o::HyperGraph::VertexIDMap::iterator v_it = optimizer->vertices().find(vid);
			if(v_it == optimizer->vertices().end())
			{
				std::cerr << "Vertex " << vid << " not in graph!" << std::endl;
				exit(-1);
			}
			g2o::VertexSBAPointXYZ * v_p = dynamic_cast< g2o::VertexSBAPointXYZ * > (v_it->second);
			vPoint[i] = v_p->estimate();
		}
	}
	
	Eigen::Vector2d predict(const Eigen::Vector3d X)
	{
		return cam_params->cam_map(X);
	}
	
	bool addCameraParams(const double f, const Eigen::Vector2d c)
	{
		cam_params = new g2o::CameraParameters (f, c, 0.);
		cam_params->setId(0);
		return optimizer->addParameter(cam_params);
	}
	
	int addPoseVertex(const g2o::SE3Quat &pose)
	{
		int pose_id = getNewUniqueId();
		
		g2o::VertexSE3Expmap *v_se3 = new g2o::VertexSE3Expmap();
		v_se3->setId(pose_id);
		v_se3->setEstimate(pose);
		
		optimizer->addVertex(v_se3);
		
		vPose.push_back(pose);
		PoseVectorId_to_VertexId[vPose.size()-1] = pose_id;
		
		return pose_id;
	}
	
	int addPointVertex(const Eigen::Vector3d &point)
	{
		int point_id = getNewUniqueId();
		
		g2o::VertexSBAPointXYZ * v_p = new g2o::VertexSBAPointXYZ();
		v_p->setId(point_id);
		v_p->setMarginalized(true);
		v_p->setEstimate(point);
		
		optimizer->addVertex(v_p);
		
		vPoint.push_back(point);
		PointVectorId_to_VertexId[vPoint.size()-1] = point_id;
		
		return point_id;
	}
	
	Eigen::Vector3d getPointFromGraph(int point_id)
	{
		g2o::HyperGraph::VertexIDMap::iterator v_it = optimizer->vertices().find(point_id);
		
		if (v_it==optimizer->vertices().end()){
	      std::cerr << "Vertex " << point_id << " not in graph!" << std::endl;
			return Eigen::Vector3d(0,0,0);
	    }
	
		g2o::VertexSBAPointXYZ * v_p
	        = dynamic_cast< g2o::VertexSBAPointXYZ * > (v_it->second);
	
		return v_p->estimate();
	}
	
	Eigen::Vector3d getPointFromResult(int idx)
	{
		return vPoint[idx];
	}
	
	void setConstraint(const int point_id,const int pose_id, const Eigen::Vector2d &z)
	{
		int from = PointVectorId_to_VertexId[point_id];
		int to = PoseVectorId_to_VertexId[pose_id];
		
		//std::cout << "From : " << from << " To : " << to << std::endl;
		
		g2o::EdgeProjectXYZ2UV * e = new g2o::EdgeProjectXYZ2UV();
		e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer->vertices().find(from)->second));
		e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer->vertices().find(to)->second));
		e->setMeasurement(z);
		e->information() = Eigen::Matrix2d::Identity();
		
		if (robustKernel)
		{
			g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
			e->setRobustKernel(rk);
		}
		
		e->setParameterId(0, 0);
		optimizer->addEdge(e);
	}
	
	int numPose()
	{
		return vPose.size();
	}
	
	int numPoint()
	{
		return vPoint.size();
	}
	
private:
	int getNewUniqueId()
	{
		++unique_id;
		return unique_id;
	}
	
private:
	//! Just some variables
	int unique_id;	//! Unique id used in construction of graph nodes.
	bool robustKernel;
	g2o::CameraParameters * cam_params;
	
	PoseVector vPose;
	PointVector vPoint;
	std::tr1::unordered_map<int,int> PoseVectorId_to_VertexId;
	std::tr1::unordered_map<int,int> PointVectorId_to_VertexId;
	
	//! g2o related variables
	g2o::SparseOptimizer *optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
	g2o::BlockSolver_6_3 *solver_ptr;
	g2o::OptimizationAlgorithmLevenberg *solver;
};