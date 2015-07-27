//
//  A super basic implementation of MCMC Metropolis Hasting for solving sudoku.
//
//  Created by Francis Engelmann on 26/07/15.
//  Copyright (c) 2015 Francis Engelmann. All rights reserved.
//

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// C/C++ includes
#include <iostream>
#include <cmath>
#include <random>
#include <map>

// PARAMETERS
unsigned int max_iterations = 1000000; // The number of iterations that the MCMC runs.

// The number of samples, used for normalizing the estimated distribution.
unsigned int sample_count = 1;
std::map<int, int> estimtated_distribution;

int field[9][9];

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

int cost() {
  int cost = 0;

  // Iterate over each cell in the field
  for (int y=0; y<9; y++) {
    for (int x=0; x<9; x++) {
      int current_cell = field[y][x];
      // In each row current cell must be unique
      for (int l=0; l<9; l++) if (current_cell == field[y][l] && x!=l) cost++;
      // In each column current cell must be unique
      for (int l=0; l<9; l++) if (current_cell == field[l][x] && y!=l) cost++;
      // In each 3x3-block current cell must be unique
      int m = x/3;
      int n = y/3;
      std::cout << m << " " << n << std::endl;
      for (int l=0; l<3; l++) {
        for (int l=0; l<3; l++) {

        }
      }
    }
  }
  return cost;
}

void show_state() {
  double m = 0; // outer margin size
  double s = 25; // width of cell
  cv::Mat image(s*9+2*m,s*9+2*m,CV_32FC3,cv::Scalar(255,255,255));
  cv::rectangle(image, cv::Point(m,m), cv::Point(s*9+m,s*9+m), cv::Scalar(0,0,0), 2);
  for (int j=0; j<10; j++) {
    for (int i=0; i<10; i++) {
      if (i%3==0 && j%3==0) {
        cv::rectangle(image, cv::Point((j*s),(i*s)), cv::Point(s*3,s*3), cv::Scalar(0,0,0), 2);
      }
      cv::rectangle(image, cv::Point((j*s),(i*s)), cv::Point(s,s), cv::Scalar(0,0,0), 1);
      cv::putText(image, std::to_string(field[i][j]), cv::Point((j*s)+5,(i*s)+s-6),
          cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,0), 1, CV_AA);

    }
  }
  cv::imshow("field",image);
  cv::waitKey(0);
}

void init_field () {
  // Setting up random number generator
  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(1,10);
  for (int j=0; j<9; j++) {
    for (int i=0; i<9; i++) {
      field[i][j] = (int)uniform_density(generator);
    }
  }
}

int main(int argc, const char * argv[]) {

  // Setting up random number generator
  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(0,1);

  init_field();
  std::cout << "Cost: " << cost() << std::endl;
  show_state();

  return 0;
  // Specify initial state
  float current_state = 7;

  // Start iterating
  for (int it=0; it<max_iterations; it++){
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
