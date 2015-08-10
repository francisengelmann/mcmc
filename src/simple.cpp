//
//  A super basic implementation of MCMC Metropolis Hasting sampling method.
//
//  Created by Francis Engelmann on 21/07/15.
//  Copyright (c) 2015 Francis Engelmann. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <random>
#include <map>

// PARAMETERS
float initial_state = 7; // Initial state of markov chain.
unsigned int max_iterations = 1000000; // Number of MCMC iterations.

// Number of samples, used for normalizing the estimated distribution.
unsigned int sample_count = 1;
std::map<int, int> estimtated_distribution;

// Mixture of two Gaussians - this will be our "hard-to-sample-from" target distribution.
float target_distribution(float x) {
  float sigma2 = 3;
  float mean2 = -15;
  float sigma1 = 5;
  float mean1 = -5;
  float pi = M_PI;
  return 0.5f/std::sqrt(2*pi)*std::pow(M_E,-0.5* ((x-mean1)/sigma1) * ((x-mean1)/sigma1) ) +
          0.5f/std::sqrt(2*pi)*std::pow(M_E,-0.5* ((x-mean2)/sigma2) * ((x-mean2)/sigma2) );
}

// Here we plot the current estimation of the target distribution
void plot_estimtated_distribution(int it) {
  double scaling = 200;
  std::cout << std::endl << "Estimated distribution - It. #" << it << std::endl;
  for (auto x : estimtated_distribution) {
    // Print the normalized estimated distribution
    std::cout << x.first << "\t" << std::string(std::round(x.second*scaling/sample_count),'*') << std::endl;
  }
}

int main(int argc, const char * argv[]) {

  // Setting up random number generator
  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(0,1);

  // Specify initial state
  float current_state = initial_state;
  
  // Start iterating
  for (unsigned int it=0; it<max_iterations; it++){
    std::normal_distribution<float> proposal_density(current_state,3); // Normal distribution at current state, mean=1
    float new_state;
    float tentative_new_state = proposal_density(generator); // Draw sample from proposal density
    float a = target_distribution(tentative_new_state) / target_distribution(current_state); // Compute acceptance rate
    if(a >= 1) {
      new_state = tentative_new_state;
    } else { // Accept new state with probability a
      if (a >= uniform_density(generator)){
        new_state = tentative_new_state;
      } else {
        new_state = current_state;
      }
    }
    ++estimtated_distribution[std::round(new_state)];
    
    // Only print a few in-between estimations
    if (it%(max_iterations/10)==0) plot_estimtated_distribution(it);
    current_state = new_state;
    sample_count++;
  }
  std::cout << "DONE" << std::endl;
  return 0;
}
