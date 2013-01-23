#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>

#include <tr1/unordered_map>

#include <unsupported/Eigen/MatrixFunctions>

//! Sophus.
#include <sophus/se3.h>

namespace Hyon
{
	
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
			
			//! Generate pose with random attitude and translation
			//true_poses_se3.push_back(Sophus::SE3(Sophus::SO3::exp(Eigen::Vector3d(uniform(), uniform(), uniform())),Eigen::Vector3d(i*0.04-1.0,0,0)));
			true_poses_se3.push_back(Sophus::SE3(pose.rotation().toRotationMatrix(),trans));
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
	
	Sophus::SE3 getPoseSE3(int idx)
	{
		return true_poses_se3.at(idx);
	}
	
private:
	std::vector<Eigen::Vector3d> true_points;
	std::vector<g2o::SE3Quat,Eigen::aligned_allocator<g2o::SE3Quat> > true_poses;
	std::vector<Sophus::SE3> true_poses_se3;
	
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


struct Point3d
{
public:
	Point3d() : id(-1), isLive(false), vertexNodeId(-1), isInsertedIntoBundler(false){}

	//! copy constructor for STL container
	Point3d(const Point3d& p)
	{
		id = p.id;
		coordinates = p.coordinates;
		coordinates_homog = p.coordinates_homog;
		isLive = p.isLive;
		vertexNodeId = p.vertexNodeId;
		isInsertedIntoBundler = p.isInsertedIntoBundler;
	}
	
	void setPoint(int pid,const Eigen::Vector3d &p)
	{
		id = pid;
		coordinates = p;
		isLive = false;
		coordinates_homog(0) = p(0);
		coordinates_homog(1) = p(1);
		coordinates_homog(2) = p(2);
		coordinates_homog(3) = 1.0;
	}
	
	Eigen::Vector4d getHomogeneous()
	{
		return coordinates_homog;
	}
	
	Eigen::Vector3d getPoint()
	{
		return coordinates;
	}
	
	void InsertBundler(int id)
	{
		vertexNodeId = id;
		isInsertedIntoBundler = true;
	}
	
	int g2oNodeId()
	{
		return vertexNodeId;
	}

private:
	//! descriptor (e.g., BRIEF, SIFT)
	// TODO: descriptor should be inserted.

	//! Location
	Eigen::Vector3d coordinates;
	Eigen::Vector4d coordinates_homog;

	//! Identification number (this should be unique throught map)
	int id;
	int vertexNodeId;
	bool isInsertedIntoBundler;
	bool isLive;
};

struct Point2d
{
public:
	Point2d(){}
	
	//! copy constructor for STL container
	Point2d(const Point2d& p)
	{
		id = p.id;
		coordinates = p.coordinates;
	}
	
	//! Image coordinates
	Eigen::Vector2d coordinates;
	int id;
};

class Map
{
public:
	typedef std::tr1::unordered_map<int,Point3d> MapVector;
	
	Map() : pointId(-1) {};
	
	//! register point to a map
	void Register(const Eigen::Vector3d &point)
	{
		int pid = getUniquePointId();
		
		Point3d p;
		p.setPoint(pid, point);
		
		//! Register into map vector
		points[pid] = p;
	};
	
	void Remove(int pid)
	{
		points.erase(pid);
	};
	
	int numPoints()
	{
		return points.size();
	}
	
	//! expensive operation?
	MapVector getMap()
	{
		return points;
	}
	
	MapVector& getMapReference()
	{
		return points;
	}
	
	Eigen::Vector3d get3Dpt(int idx)
	{
		return points[idx].getPoint();
	}
	
private:
	//! return unique point idenfication number
	int getUniquePointId()
	{
		++pointId;
		return pointId;
	}
	
	//! point index. always increasing.
	int pointId;
	
	//! vector of 3d points
	MapVector points;
};

//! Camera simulator.
class Camera
{
public:
	typedef std::tr1::unordered_map<int,Point2d> ImageVector;
	typedef std::tr1::unordered_map<int,Point3d> MapVector;
	
	Camera(double f, const Eigen::Vector2d &c) : focal_distance(f), principal_point(c)
	{
		//! Construct K matrix from f,c
		K.setIdentity();
		K(0,0) = f; 
		K(1,1) = f;
		K.col(2).head(2) = c;
	}
	
	~Camera(){}
	
	ImageVector Project(const Sophus::SE3 pose, MapVector points)
	{
		ImageVector z;
		Eigen::Matrix<double,3,4> P;	//! camera matrix
		
		P.block(0,0,3,3) = pose.rotation_matrix();
		P.col(3).head(3) = pose.translation();
		P = K*P;
		
		//! project all 3d points
		for ( auto it = points.begin(); it != points.end(); ++it )
		{
			// it->first : map id
			// it->second : coordinates
			Eigen::Vector3d uv_homog = P*it->second.getHomogeneous();
			
			//! dehomonize
			Eigen::Vector2d uv;
			uv(0) = uv_homog(0) / uv_homog(2);
			uv(1) = uv_homog(1) / uv_homog(2);
			
			//! Noise
			uv += Eigen::Vector2d(Sample::gaussian(1.0),Sample::gaussian(1.0));
			
			Point2d p;
			p.id = it->first;
			p.coordinates = uv;
			
			z[it->first] = p;
		}
		
		//! check points are in view
		for ( auto it = z.begin(); it != z.end(); ++it )
		{
			if(it->second.coordinates[0] >= 0 && it->second.coordinates[1] >=0 && it->second.coordinates[0] < 640 && it->second.coordinates[1] < 480)
			{
				continue;
			}
			
			z.erase(it);
		}
		
		return z;
	}
private:
	Eigen::Matrix3d K;
	double focal_distance;
	Eigen::Vector2d principal_point;
};

//! discrete motion data structure
struct KeyFrame
{
public:
	typedef std::tr1::unordered_map<int,Point2d> ImageVector;
	
	KeyFrame(Sophus::SE3 _pose, ImageVector observations_): pose(_pose), observations(observations_), vertexNodeId(-1), isInsertedIntoBundler(false){};
	~KeyFrame(){};
	
	/// Copy con
	KeyFrame(const KeyFrame &others)
	{
		pose = others.pose;
		observations = others.observations;
		vertexNodeId = others.vertexNodeId;
		isInsertedIntoBundler = others.isInsertedIntoBundler;
	}
	
	void InsertBundler(int id)
	{
		vertexNodeId = id;
		isInsertedIntoBundler = true;
	}
	
	Sophus::SE3 getPose()
	{
		return pose;
	}
	
	ImageVector getObservations()
	{
		return observations;
	}
	
	int g2oNodeId()
	{
		return vertexNodeId;
	}
	
private:
	//! Pose
	Sophus::SE3 pose;
	
	//! Measurements
	ImageVector observations;
	
	int vertexNodeId;
	bool isInsertedIntoBundler;
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
	
	int push_pose(const g2o::SE3Quat &pose)
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
	
	int push_pose(const Sophus::SE3 &pose)
	{
		int pose_id = getNewUniqueId();
		
		g2o::VertexSE3Expmap *v_se3 = new g2o::VertexSE3Expmap();
		v_se3->setId(pose_id);
		
		g2o::SE3Quat pose_g2o(pose.unit_quaternion(),pose.translation());
		v_se3->setEstimate(pose_g2o);
		optimizer->addVertex(v_se3);
		
		vPose.push_back(pose_g2o);
		
		PoseVectorId_to_VertexId[vPose.size()-1] = pose_id;
		
		return pose_id;
	}
	
	int push_point(const Eigen::Vector3d &point)
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
		//int from = PointVectorId_to_VertexId[point_id];
		//int to = PoseVectorId_to_VertexId[pose_id];
		int from = point_id;
		int to = pose_id;
		
		g2o::EdgeProjectXYZ2UV *e = new g2o::EdgeProjectXYZ2UV();
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

class SceneManager
{
public:
	typedef std::tr1::unordered_map<int,Point2d> ImageVector;
	typedef std::tr1::unordered_map<int,Point3d> MapVector;
	
	SceneManager(double f, const Eigen::Vector2d &c)
	{
		map = new Map();
		ground_truth_map = new Map();
		camera = new Camera(f,c);
		bundler = new GraphBundler(false, /* dense? */
							true /* robust kern? */);
							
		if (!bundler->addCameraParams(f,c))
		{
			assert(false);
		}
	}
	
	~SceneManager()
	{
		if(map != NULL)
			delete map;
			
		if(camera != NULL)
			delete camera;
			
		if(ground_truth_map != NULL)
			delete ground_truth_map;
	}
	
	void pushPose(const Sophus::SE3 pose)
	{
		ImageVector z = camera->Project(pose,map->getMap());
		KeyFrame kf(pose,z);
		vKeyFrame.push_back(kf);
		
		//! TODO : Insert outlier information
	}
	
	void pushPoint(const Eigen::Vector3d &point)
	{
		//! insert into map
		map->Register(point);
	}
	
	void pushGroundTruth(const Eigen::Vector3d &point)
	{
		ground_truth_map->Register(point);
	}
	
	void bundle()
	{
		//! fetch all available map and insert into bundler
		MapVector &landmarks = map->getMapReference();
		
		for ( auto it = landmarks.begin(); it != landmarks.end(); ++it )
		{
			//! insert into bundler
			Point3d &p = it->second;
			p.InsertBundler(bundler->push_point(it->second.getPoint()));
		}
		
		int point_num = 0;
		double sum_diff2 = 0;
		
		for(size_t i=0;i<bundler->numPoint();++i)
		{
			Eigen::Vector3d diff = bundler->getPointFromResult(i) - ground_truth_map->get3Dpt(i);
			sum_diff2 += diff.dot(diff);
			++point_num;
		}
		
		std::cout << "Point error before optimisation (inliers only): " << sqrt(sum_diff2/point_num) << std::endl;
		
		//! fetch all pose and insert into bundler
		for ( auto it = vKeyFrame.begin(); it != vKeyFrame.end(); ++it)
		{
			KeyFrame &kf = *it;
			kf.InsertBundler(bundler->push_pose(kf.getPose()));
			
			//! set relation based on keyframe information
			ImageVector z = kf.getObservations();
			
			if(z.size() >= 2)
			{
				for ( auto it = z.begin(); it != z.end(); ++it )
				{
					//! get 3d pt id (which 3d point was observed?)
					int point_vertex_id = landmarks[it->first].g2oNodeId();
					
					bundler->setConstraint(point_vertex_id,kf.g2oNodeId(),it->second.coordinates);
				}
			}
		}
		
		bundler->doBundleAdjustment(25);
		
		sum_diff2 = 0;
		point_num = 0;
		
		//! for statistics		
		for(size_t i=0;i<bundler->numPoint();++i)
		{
			Eigen::Vector3d diff = bundler->getPointFromResult(i) - ground_truth_map->get3Dpt(i);
			sum_diff2 += diff.dot(diff);
			++point_num;
		}
		
		std::cout << "Point error after optimisation (inliers only): " << sqrt(sum_diff2/point_num) << std::endl;
	}
	
private:
	//! Vector of keyframe
	std::vector<KeyFrame> vKeyFrame;
	Map *map;
	Map *ground_truth_map;
	Camera *camera;
	GraphBundler *bundler;
};

};