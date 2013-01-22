#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>

#include <tr1/unordered_set>
#include <tr1/unordered_map>
#include <tr1/memory>

class GraphBundler
{
public:
	//! Data structures
	typedef std::tr1::unordered_map<int,g2o::SE3Quat> poseVertices;
	typedef std::tr1::unordered_map<int,g2o::VertexSBAPointXYZ> pointVertices;
	
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
	
	~GraphBundler()
	{
	}
	
	void doBundleAdjustment(int numIterations)
	{
		optimizer->initializeOptimization();
		optimizer->setVerbose(true);
		optimizer->optimize(numIterations);
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
		
		//poseVertices[pose_id] = pose;
		
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
		
		//pointVertices[point_id] = v_p;
		
		return point_id;
	}
	
	void setConstraint(const int from,const int to, const Eigen::Vector2d &z)
	{
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
	
	//! g2o related variables
	g2o::SparseOptimizer *optimizer;
	g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
	g2o::BlockSolver_6_3 *solver_ptr;
	g2o::OptimizationAlgorithmLevenberg *solver;
};