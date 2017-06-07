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

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  // Select number of particles to use
  num_particles = 2;

  // Set up normal distributions
  default_random_engine generator;
  normal_distribution<double> x_dist(x,std[0]);
  normal_distribution<double> y_dist(y,std[1]);
  normal_distribution<double> theta_dist(theta,std[2]);

  cout << "in init" << endl;
  cout << "x: " << x << " y: " << y << " theta: " << theta << endl;

  // Initialize each particle with random noise
  for(int i = 0; i < num_particles; i++) {
    // Create a particle randomly distributed about the initial estimated postion
    Particle particle;
    particle.id = i;
    particle.x = x_dist(generator);
    particle.y = y_dist(generator);
    particle.theta = theta_dist(generator);
    particle.weight = 1.0;
    weights.push_back(1.0);

    // Append particle to particles
    particles.push_back(particle);

    cout << "curr x: " << particles[i].x << " curr y: " << particles[i].y << " curr theta: " << particles[i].theta << endl;
  }

  // Filter is now initialized
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


  // Set up normal distributions
  default_random_engine generator;
  normal_distribution<double> x_dist(0,std_pos[0]);
  normal_distribution<double> y_dist(0,std_pos[1]);
  normal_distribution<double> theta_dist(0,std_pos[2]);

  cout << "in prediction" << endl;
  cout << "delta_t: " << delta_t << " std_pos: " << std_pos[0] << " " << std_pos[1] << " " << std_pos[2] << " velocity: " << velocity << " yaw rate: " << yaw_rate << endl;

  // Initialize each particle with random noise
  for(int i = 0; i < num_particles; i++) {

    cout << "x: " << particles[i].x << " y: " << particles[i].y << " theta: " << particles[i].theta << endl;

    if (fabs(yaw_rate) < 0.00001) {  
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else { 

      double theta_next = particles[i].theta + yaw_rate*delta_t;
      double a = velocity/yaw_rate;
      particles[i].x = particles[i].x + a * (sin(theta_next) - sin(particles[i].theta));
      particles[i].y = particles[i].y + a * (-cos(theta_next) + cos(particles[i].theta));
      particles[i].theta = theta_next;
  }

    // Create a particle randomly distributed about the initial estimated postion
    particles[i].x += x_dist(generator);
    particles[i].y += y_dist(generator);
    particles[i].theta += theta_dist(generator);

    cout << "curr x: " << particles[i].x << " curr y: " << particles[i].y << " curr theta: " << particles[i].theta << endl;
  }

}

void ParticleFilter::dataAssociation(std::vector<Map::single_landmark_s> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

  cout << "data association" << endl;
  double min_distance;
  for(int i = 0; i < observations.size(); i++) {
   // cout << "observation " << i << endl;
    for(int j = 0; j < predicted.size(); j++) {
   //   cout << "predicted " << j << endl;
      double distance = landmark_dist(predicted[j], observations[i]);
      if (j == 0) min_distance = distance;
      if (distance <= min_distance) {
        min_distance = distance;
        observations[i].id = j;
      }
    }
  }
}

inline double gauss2(Map::single_landmark_s predicted, LandmarkObs observations, double std[])
{

  double xerr = (observations.x - predicted.x_f);
  double yerr = (observations.y - predicted.y_f);
  double num = exp(-0.5*(xerr*std[0]*std[0]*xerr + yerr*std[1]*std[1]*yerr));
  return num/2.0*M_PI*std[0]*std[1];
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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

  cout << "start updateWeights" << endl;

  for(int i = 0; i < num_particles; i++) {
    std::vector<LandmarkObs> observations_map_frame;
    cout << "particle: " << i << endl;

    for(int i = 0; i < observations.size(); i++) {
      double x = observations[i].x*cos(particles[i].theta) - observations[i].y*sin(particles[i].theta) + particles[i].x;
      double y = observations[i].x*sin(particles[i].theta) + observations[i].y*cos(particles[i].theta) + particles[i].y;
      observations[i].x = x;
      observations[i].y = y;
      observations_map_frame.push_back(observations[i]);
    }
    dataAssociation(map_landmarks.landmark_list, observations_map_frame);
    cout << "data assoc complete" << endl;
    double weight = 1;
    for(int i = 0; i < observations.size(); i++) {
      Map::single_landmark_s predicted = map_landmarks.landmark_list[observations[i].id];
      weight *= gauss2(predicted, observations[i], std_landmark);
    }
    cout << "gauss complete" << endl;
    particles[i].weight = weight;
    cout << "weight: " << weight << endl;
    weights[i] = weight;
    cout << "update particle " << i << " complete" << endl;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  cout << "start resample" << endl;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> start_idx_distribution(0,num_particles-1);

  int idx = start_idx_distribution(generator);

  double max_weight = *max_element(begin(weights), end(weights));
  std::uniform_real_distribution<double> beta_distribution(0.0,2*max_weight);
  double beta = 0;
  std::vector<Particle> resampled_particles;
  cout << "Start idx: "  << idx << " max weight: " << max_weight << endl;
  for(int i = 0; i < num_particles; i++) {
    beta += beta_distribution(generator);
    cout << "resample step: " << i << " beta start: " << beta << " idx start: " << idx << endl;
    while (beta > weights[idx]) {
      beta -= weights[idx];
      idx = (idx + 1) % num_particles;
    }
    cout << "beta end: " << beta << " idx end: " << idx << endl;
    resampled_particles.push_back(particles[idx]);
    cout << "push back complete" << endl;
  }
  cout << "indexing complete" << endl;
  particles = resampled_particles;
  cout << "resample complete" << endl;
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
