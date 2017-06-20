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
	//  Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

  // Select number of particles to use
  // Surpisingly the filter will pass with as few as 7 particles! 
  // 100 particles performs well in speed and accuracy
  num_particles = 100;

  // Set up normal distributions
  default_random_engine generator;
  normal_distribution<double> x_dist(x,std[0]);
  normal_distribution<double> y_dist(y,std[1]);
  normal_distribution<double> theta_dist(theta,std[2]);

  // Initialize each particle with random noise
  for(int i = 0; i < num_particles; i++) {
    // Create a particle randomly distributed about the initial estimated postion
    Particle particle;
    particle.id = i;
    particle.x = x_dist(generator);
    particle.y = y_dist(generator);
    particle.theta = theta_dist(generator);

    // Initialize all weights to 1
    particle.weight = 1.0;
    weights.push_back(1.0);

    // Append particle to particles
    particles.push_back(particle);

  }

  // Filter is now initialized
  is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.

  // Set up normal distributions
  default_random_engine generator;
  normal_distribution<double> x_dist(0,std_pos[0]);
  normal_distribution<double> y_dist(0,std_pos[1]);
  normal_distribution<double> theta_dist(0,std_pos[2]);

  // Calculate predictions for each particle using bicycle model
  for(int i = 0; i < num_particles; i++) {

    if (fabs(yaw_rate) < 0.00001) {  
      // Handle yaw_rate = 0 case
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
    } else { 
      // bicycle model
      double theta_next = particles[i].theta + yaw_rate*delta_t;
      double a = velocity/yaw_rate;
      particles[i].x = particles[i].x + a * (sin(theta_next) - sin(particles[i].theta));
      particles[i].y = particles[i].y + a * (-cos(theta_next) + cos(particles[i].theta));
      particles[i].theta = theta_next;
  }

    // Add noise to particles
    particles[i].x += x_dist(generator);
    particles[i].y += y_dist(generator);
    particles[i].theta += theta_dist(generator);

  }

}

/*
 * Computes the Euclidean distance between two 2D points.
 * @param (x1,y1) x and y coordinates of first point
 * @param (x2,y2) x and y coordinates of second point
 * @output Euclidean distance between two 2D points
 */
inline double landmark_dist(LandmarkObs a, LandmarkObs b) {
  return dist(a.x, a.y, b.x, b.y);
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) { 

	//  Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.

  // Loop through each observation
  for(int i = 0; i < observations.size(); i++) {
    // For each observation we need to check all of the predict landmarks
    // and find the one with minimum distance
    double min_distance;
    for(int j = 0; j < predicted.size(); j++) {
      // Calculate distance between observation and landmark
      double distance = landmark_dist(predicted[j], observations[i]);

      // Keep track of the minimum distance and set the id of the observation
      // to the prediction of minimum distance. This is the association
      if (j == 0) min_distance = distance;
      if (distance <= min_distance) {
        min_distance = distance;
        observations[i].id = j;
      }
    }
  }
}

inline double gauss2(LandmarkObs predicted, LandmarkObs observation, double varx, double vary, double den)
{
  // This funciton performs a 2d gaus function between an observation and the associated
  // landmark with covariance std

  double dist = landmark_dist(predicted, observation);
  double xerr = (observation.x - predicted.x);
  double yerr = (observation.y - predicted.y);
  double num = exp(-0.5*(xerr*varx*xerr + yerr*vary*yerr));
  double out = num/den;
  return out;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution

  // Populate prediced with vector of landmakr objects from the Map
  std::vector<LandmarkObs> predicted;
  for(int i = 0; i < map_landmarks.landmark_list.size(); i++) {
      LandmarkObs landmark;
      landmark.x = map_landmarks.landmark_list[i].x_f;
      landmark.y = map_landmarks.landmark_list[i].y_f;
      landmark.id = map_landmarks.landmark_list[i].id_i;
      predicted.push_back(landmark);
  }

  // Pre compute constants for use in gauss2 weighting function
  double varx = std_landmark[0]*std_landmark[0];
  double vary = std_landmark[1]*std_landmark[1];
  double den = 2.0*M_PI*std_landmark[0]*std_landmark[1];

  // Loop through particles
  for(int i = 0; i < num_particles; i++) {
    // Create a vector of Landmark Objects that will be in the ma frame
    std::vector<LandmarkObs> observations_map_frame;

    // Convert all of the observations to the map frame for this particle
    for(int j = 0; j < observations.size(); j++) {
      double x = observations[j].x*cos(particles[i].theta) - observations[j].y*sin(particles[i].theta) + particles[i].x;
      double y = observations[j].x*sin(particles[i].theta) + observations[j].y*cos(particles[i].theta) + particles[i].y;
      LandmarkObs observation;
      observation.x = x;
      observation.y = y;
      observation.id = observations[j].id;
      observations_map_frame.push_back(observation);
    }

    // Associate the observations in the map frame with the predicted landmarks
    dataAssociation(predicted, observations_map_frame);

    // Calculate the weight for each particle using 2d gauss function
    double weight = 1;
    for(int i = 0; i < observations_map_frame.size(); i++) {
      LandmarkObs landmark = predicted[observations_map_frame[i].id];
      weight *= gauss2(landmark, observations_map_frame[i], varx, vary, den);
    }

    // Assign weights
    particles[i].weight = weight;
    weights[i] = weight;
  }
}

void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 

  // Select random starting particle
  std::default_random_engine generator;
  std::uniform_int_distribution<int> start_idx_distribution(0,num_particles-1);
  int idx = start_idx_distribution(generator);

  // Get max weight value for setting beta distribution limit
  double max_weight = *max_element(begin(weights), end(weights));

  // Set up uniform distribution for beta
  std::uniform_real_distribution<double> beta_distribution(0.0,2*max_weight);

  // Wheel method for resampling
  double beta = 0;
  std::vector<Particle> resampled_particles;
  for(int i = 0; i < num_particles; i++) {
    beta += beta_distribution(generator);
    while (beta > weights[idx]) {
      beta -= weights[idx];
      idx = (idx + 1) % num_particles;
    }
    resampled_particles.push_back(particles[idx]);
  }
  particles = resampled_particles;
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
