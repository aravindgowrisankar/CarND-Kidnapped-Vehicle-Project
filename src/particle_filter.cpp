/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>
#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std_devs[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
  
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  num_particles=100;
  normal_distribution<double> dist_x(x, std_devs[0]);
  normal_distribution<double> dist_y(y, std_devs[1]);
  normal_distribution<double> dist_theta(theta, std_devs[2]);
  default_random_engine gen;

  for (int i=0;i<num_particles;i++) {
    Particle p;
    p.id=i;
    p.x=dist_x(gen);
    p.y=dist_y(gen);
    p.theta=dist_theta(gen);
    p.weight=1.0;
    weights.push_back(1.0);//Default weight for the vector in Particle filter
    particles.push_back(p);
  }
  is_initialized=true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
  // TBD: Adding noise

  for (int i=0;i<num_particles;i++) {
    double x=particles[i].x;
    double y=particles[i].y;
    double theta=particles[i].theta;
    double theta_new=theta+(yaw_rate*delta_t);
    double x_new=x+(velocity/yaw_rate)*(sin(theta_new)-sin(theta));
    double y_new=y+(velocity/yaw_rate)*(cos(theta)-cos(theta_new));
    normal_distribution<double> dist_x(x_new, std_pos[0]);
    normal_distribution<double> dist_y(y_new, std_pos[1]);
    normal_distribution<double> dist_theta(theta_new, std_pos[2]);
    default_random_engine gen;

    particles[i].theta=dist_theta(gen);
    particles[i].x=dist_x(gen);
    particles[i].y=dist_y(gen);
    //cout<<"Old:("<<x<<","<<y<<") New:("<<particles[i].x<<","<<particles[i].y;
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  double num_predicted=predicted.size();
  double num_observations=observations.size();
  int min_j;
  //cout<<"Num Landmarks"<<num_predicted<<" Num observations"<<num_observations;
  if (num_predicted==0)
    return;
  double min_dist;
  int landmark_index;
  for (int i=0;i<observations.size();i++) {
    //cout<<"Original observation ID "<<observations[i].id;
    min_dist=std::numeric_limits<double>::max();
    for (int j=0;j<predicted.size();j++) {
      double distance=dist(observations[i].x,observations[i].y,predicted[j].x,predicted[j].y);
      //cout<<"Distance between observation "<<observations[i].id<<" and landmark "<<predicted[j].id<<"="<<distance<<endl;
      if (distance<min_dist) {
        min_dist=distance;
        min_j=j;
      }
      
    }
    observations[i].id=predicted[min_j].id;
    //cout<<"Assigned obs("<<observations[i].x<<","<<observations[i].y<<") to landmark"<<observations[i].id<<". ("<<predicted[min_j].x<<","<<predicted[min_j].y<<")"<<endl;

    //cout<<"Landmark for observation  "<<observations[i].id<<endl;
  }
  //cout<<"End of data associaton"<<endl;

}

std::vector<LandmarkObs> transform_old(double particle_x, double particle_y, double  particle_theta, std::vector<LandmarkObs> observations) {
  std::vector<LandmarkObs> transformed_observations;
  for (int i=0;i<observations.size();i++) {
    LandmarkObs tobs;
    tobs.id=observations[i].id;
    double r=observations[i].y-(observations[i].x/tan(particle_theta));
    tobs.x=particle_x-(r*sin(particle_theta));
    tobs.y=particle_y+(observations[i].x/sin(particle_theta))+(r*cos(particle_theta));
    transformed_observations.push_back(tobs);
    //cout<<"original ("<<observations[i].x<<","<<observations[i].y<<")"<<" transformed ("<<tobs.x<<","<<tobs.y<<")";
  }
  return transformed_observations;
}

std::vector<LandmarkObs> transform(double particle_x, double particle_y, double  particle_theta, std::vector<LandmarkObs> observations) {
  std::vector<LandmarkObs> transformed_observations;
  for (int i=0;i<observations.size();i++) {
    LandmarkObs tobs;
    tobs.id=observations[i].id;
    tobs.x=particle_x+(observations[i].x*cos(particle_theta))-(observations[i].y*sin(particle_theta));
    tobs.y=particle_y+(observations[i].x*sin(particle_theta))+(observations[i].y*cos(particle_theta));
    transformed_observations.push_back(tobs);
    //cout<<"original ("<<observations[i].x<<","<<observations[i].y<<")"<<" transformed ("<<tobs.x<<","<<tobs.y<<")";
  }
  return transformed_observations;
}

double update_particle_weight(std::vector<LandmarkObs> observations,Map map_landmarks){
  double prob = 1.0;

  double sigma_pos [3] = {0.3, 0.3, 0.01}; // GPS measurement uncertainty [x [m], y [m], theta [rad]]
  double gaussian_dr=2*M_PI*sigma_pos[0]*sigma_pos[1];
  double sigma_x_sq=sigma_pos[0]*sigma_pos[0];
  double sigma_y_sq=sigma_pos[1]*sigma_pos[1];

  for (int i=0;i<observations.size();i++) {
    int map_index=observations[i].id-1;
    
    double x=map_landmarks.landmark_list[map_index].x_f;
    double y=map_landmarks.landmark_list[map_index].y_f;
    int id=map_landmarks.landmark_list[map_index].id_i;
    if (id!=observations[i].id)
      cout<<"Exception Map id"<<id<<" is not the same as tobs id"<<observations[i].id;
    double x_comp=-(observations[i].x-x)*(observations[i].x-x)/(2*sigma_x_sq);
    double y_comp=-(observations[i].y-y)*(observations[i].y-y)/(2*sigma_y_sq);
    //cout<<"Map index"<<map_index<<"location "<<x<<","<<y<<"observation "<<observations[i].x<<","<<observations[i].y;
    prob *= exp(x_comp+y_comp)/gaussian_dr;
  }
  //cout<<"Weight"<<prob;
  return prob;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {

  cout<<"Obs size"<<observations.size();
  cout<<"Sensor range"<<sensor_range<<endl;
  for (int i=0;i<num_particles;i++) {
    std::vector<LandmarkObs> transformed_obs=transform(particles[i].x, particles[i].y, particles[i].theta,observations);
    std::vector<LandmarkObs> map_landmarks_subset;

    for (int i=0;i<map_landmarks.landmark_list.size();i++) {
      double x=map_landmarks.landmark_list[i].x_f;
      double y=map_landmarks.landmark_list[i].y_f;
      int id=map_landmarks.landmark_list[i].id_i;
      double distance=dist(particles[i].x,particles[i].y,x,y);
      //cout<<"Distance="<<distance;
      if (distance<sensor_range){
        LandmarkObs candidate;
        candidate.x=x;
        candidate.id=id;
        candidate.y=y;
        //cout<<"Adding map id "<<candidate.id<<" at location ("<<candidate.x<<","<<candidate.y<<") to subset";
        map_landmarks_subset.push_back(candidate);
      }
    }//compute landmarks
    dataAssociation(map_landmarks_subset,transformed_obs);
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;
    for (int t=0;t<transformed_obs.size();t++) {
      associations.push_back(transformed_obs[t].id);
      sense_x.push_back(transformed_obs[t].x);
      sense_y.push_back(transformed_obs[t].y);
    }

    SetAssociations(particles[i],associations, sense_x, sense_y);
    particles[i].weight=update_particle_weight(transformed_obs,map_landmarks);
    weights[i]=particles[i].weight;
  }
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to their weight. 
  // NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  double highest_weight = -1.0;
  double weight_sum = 0.0;
  for (int i = 0; i < num_particles; ++i) {
    if (particles[i].weight > highest_weight) {
      highest_weight = particles[i].weight;
    }
    weight_sum += particles[i].weight;
  }
  cout << "In Resample: before  highest w " << highest_weight << endl;
  cout << "In Resample: before  average w " << weight_sum/num_particles << endl;  


  std::random_device rd;
  std::mt19937 gen(rd());
  std::vector<Particle> new_particles;
  std::discrete_distribution<double> d(weights.begin(), weights.end());
  //weights.clear();
  //double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  //cout<<"sum="<<sum<<endl;
  for (int i=0;i<num_particles;i++){
    int selected=d(gen);
    //cout <<"selected particle"<<selected;
    Particle p;
    p.id=i;
    p.x=particles[selected].x;
    p.y=particles[selected].y;
    p.theta=particles[selected].theta;
    p.weight=particles[selected].weight;
    new_particles.push_back(p);
  }
  particles.clear();
  particles=new_particles;
  highest_weight = -1.0;
  weight_sum = 0.0;
  for (int i = 0; i < num_particles; ++i) {
    if (particles[i].weight > highest_weight) {
      highest_weight = particles[i].weight;
    }
    weight_sum += particles[i].weight;
    cout<<"Particle "<<i<<" ("<<particles[i].x<<","<<particles[i].y<<")"<<endl;
  }
  cout << "Resample highest w " << highest_weight << endl;
  cout << "Resample average w " << weight_sum/num_particles << endl;  
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
