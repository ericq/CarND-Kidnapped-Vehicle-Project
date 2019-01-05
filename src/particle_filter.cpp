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
#include <map>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[])
{

	// DONE: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 40;

	default_random_engine gen;

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; i++)
	{

		Particle p = {i, dist_x(gen), dist_y(gen), dist_theta(gen), 1.0};
		particles.push_back(p);

		// Print your samples to the terminal.
		cout << "init <" << p.id << ">" << p.x << " " << p.y << " " << p.theta << endl;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// DONE: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// cout << "entering prediction.. delta_t=" << delta_t << " std_pos=" << std_pos[0] << "," << std_pos[1]
	// 	 << " velocity=" << velocity << " yaw_rate=" << yaw_rate << endl;

	default_random_engine gen;

	for (int i = 0; i < num_particles; i++)
	{

		Particle &p = particles[i]; // !!! USE reference please

		//cout << "prediction input <" << p.id << ">" << p.x << " " << p.y << " " << p.theta << endl;

		normal_distribution<double> dist_nx, dist_ny, dist_ntheta;

		double nx, ny, ntheta;

		if (yaw_rate != 0)
		{
			nx = p.x + velocity * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) / yaw_rate;
			ny = p.y + velocity * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) / yaw_rate;
			ntheta = p.theta + yaw_rate * delta_t;
		}
		else
		{
			cout << "yaw_rate == 0" << endl;
			nx = p.x + velocity * delta_t * cos(p.theta);
			ny = p.y + velocity * delta_t * sin(p.theta);
			ntheta = p.theta;
		}

		//cout << "prediction output <" << p.id << ">" << nx << " " << ny << " " << ntheta << endl;

		dist_nx = normal_distribution<double>(nx, std_pos[0]);
		dist_ny = normal_distribution<double>(ny, std_pos[1]);
		dist_ntheta = normal_distribution<double>(ntheta, std_pos[2]);

		p.x = dist_nx(gen);
		p.y = dist_ny(gen);
		p.theta = dist_ntheta(gen);

		//cout << "prediction with normal_dist:" << p.id << ">" << p.x << " " << p.y << " " << p.theta << endl;
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	// Not needed. it's more efficient to do everything in updateWeights directly
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// DONE: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	//cout << "entering updateWeights.. " << endl;

	// for convenience - put map into a "map" data structure, key = landmark id
	map<int, Map::single_landmark_s> conv_map;
	for (auto lm : map_landmarks.landmark_list)
	{
		conv_map[lm.id_i] = lm;
	}

	// clear the weigths if it's not empty
	weights.clear();

	for (auto &p : particles)
	{
		//cout << " start particle processing: " << p.id << endl;
		vector<LandmarkObs> predicted;
		vector<LandmarkObs> observations_cp = observations;

		// store the associated result - perception of landmarks from a particle's eye
		vector<int> assoc;		// a vector to each nearest landmark id
		vector<double> assoc_x; // perception of the landmark postion in map coordinates - x
		vector<double> assoc_y; //  - y

		vector<double> shortest_dist_lm;

		// for each observation, convert its car coordindates to map coordinates
		// and find the nearest landmark
		// store the result in p as association, sense_x, sense_y eventually
		for (auto ob : observations_cp)
		{
			//
			// double dist_before_conv = sqrt(ob.x * ob.x + ob.y * ob.y);
			// if (dist_before_conv > sensor_range)
			// {
			// 	cout << " the observation distance greater than sensor range: " << dist_before_conv << " skipped." << endl;
			// 	continue;
			// }

			// convert observation from car coordinate to map coordinate using homogeneous transformation matrix
			// to further clarify:
			// the input observation_cp (input 1) is provided by the vehicle
			// we map the observation into each particl's perspective (input 2: particle's x,y,theta)
			// i.e., as if the observations are from that particle

			double x_m = p.x + cos(p.theta) * ob.x - sin(p.theta) * ob.y;
			double y_m = p.y + sin(p.theta) * ob.x + cos(p.theta) * ob.y;

			// the observations_cp now has all x,y in map coordinates
			int best_land_mark = -1;
			// use Nearest-Neighbor to find which landmark shall be used
			double shortest_dist = numeric_limits<double>::max();
			for (auto lm : map_landmarks.landmark_list)
			{
				double diffx = x_m - lm.x_f;
				double diffy = y_m - lm.y_f;
				double d = sqrt(diffx * diffx + diffy * diffy);

				if (d < shortest_dist)
				{
					shortest_dist = d;
					best_land_mark = lm.id_i;
				}
			}

			assert(best_land_mark != -1);
			shortest_dist_lm.push_back(shortest_dist);

			//
			// if (shortest_dist > sensor_range)
			// {
			// 	cout << "shortest_dist to best landmark greater than sensor range : " << shortest_dist << "skip this observation" << endl;
			// 	continue;
			// }

			assoc.push_back(best_land_mark);
			assoc_x.push_back(x_m);
			assoc_y.push_back(y_m);
		}

		// update weights
		double total_weight = 1.0; // use 1.0 not 0.0 because it will be a production, not addition

		for (int i = 0; i < assoc.size(); i++)
		{

			// calculate weight for each associated landmark
			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];
			double x_obs = assoc_x[i];
			double y_obs = assoc_y[i];
			double mu_x = conv_map[assoc[i]].x_f;
			double mu_y = conv_map[assoc[i]].y_f;
			//calculate normalization term
			double gauss_norm = (1 / (2 * M_PI * sig_x * sig_y));

			//calculate exponent
			double exponent = pow((x_obs - mu_x), 2) / (2 * pow(sig_x, 2)) + pow((y_obs - mu_y), 2) / (2 * pow(sig_y, 2));

			//calculate weight using normalization terms and exponent
			double weight = gauss_norm * exp(-exponent);

			// update total weight
			total_weight *= weight;
		}

		p.weight = total_weight;
		weights.push_back(total_weight);
		//cout << "total weight:" << total_weight << endl;
	}

	assert(weights.size() == num_particles);
}

void ParticleFilter::resample()
{
	// DONE: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// cout << "entering resample.. " << endl;
	// cout << "weights=";
	// for (auto w : weights)
	// {
	// 	cout << w << " ";
	// }
	// cout << endl;

	random_device rd;
	mt19937 gen(rd());
	discrete_distribution<> dd(weights.begin(), weights.end());
	vector<Particle> resampled;

	//cout << "start to pick: ";
	for (int i = 0; i < num_particles; i++)
	{
		int picked_pid = dd(gen);
		//cout << " " << picked_pid;
		resampled.push_back(particles[picked_pid]);
	}
	cout << endl;

	particles = resampled;
	// TODO - may need to reset weights?
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
										 const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;

	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
