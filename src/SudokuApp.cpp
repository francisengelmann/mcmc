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
      for (int l=0; l<3; l++) {
        for (int k=0; k<3; k++) {
          if (current_cell == field[3*n+l][3*m+k] && y!=3*n+l && x!=3*m+k) cost++;
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

void init_solved_field () {
  // Setting up random number generator
  field[0][0] = 8;
  field[0][1] = 2;
  field[0][2] = 7;
  field[0][3] = 1;
  field[0][4] = 5;
  field[0][5] = 4;
  field[0][6] = 3;
  field[0][7] = 9;
  field[0][8] = 6;

  field[1][0] = 9;
  field[1][1] = 6;
  field[1][2] = 5;
  field[1][3] = 3;
  field[1][4] = 2;
  field[1][5] = 7;
  field[1][6] = 1;
  field[1][7] = 4;
  field[1][8] = 8;

  field[2][0] = 3;
  field[2][1] = 4;
  field[2][2] = 1;
  field[2][3] = 6;
  field[2][4] = 8;
  field[2][5] = 9;
  field[2][6] = 7;
  field[2][7] = 5;
  field[2][8] = 2;

  field[3][0] = 5;
  field[3][1] = 9;
  field[3][2] = 3;
  field[3][3] = 4;
  field[3][4] = 6;
  field[3][5] = 8;
  field[3][6] = 2;
  field[3][7] = 7;
  field[3][8] = 1;

  field[4][0] = 4;
  field[4][1] = 7;
  field[4][2] = 2;
  field[4][3] = 5;
  field[4][4] = 1;
  field[4][5] = 3;
  field[4][6] = 6;
  field[4][7] = 8;
  field[4][8] = 9;

  field[5][0] = 6;
  field[5][1] = 1;
  field[5][2] = 8;
  field[5][3] = 9;
  field[5][4] = 7;
  field[5][5] = 2;
  field[5][6] = 4;
  field[5][7] = 3;
  field[5][8] = 5;

  field[6][0] = 7;
  field[6][1] = 8;
  field[6][2] = 6;
  field[6][3] = 2;
  field[6][4] = 3;
  field[6][5] = 5;
  field[6][6] = 9;
  field[6][7] = 1;
  field[6][8] = 4;

  field[7][0] = 1;
  field[7][1] = 5;
  field[7][2] = 4;
  field[7][3] = 7;
  field[7][4] = 9;
  field[7][5] = 6;
  field[7][6] = 8;
  field[7][7] = 2;
  field[7][8] = 3;

  field[8][0] = 2;
  field[8][1] = 3;
  field[8][2] = 9;
  field[8][3] = 8;
  field[8][4] = 4;
  field[8][5] = 1;
  field[8][6] = 5;
  field[8][7] = 6;
  field[8][8] = 7;
}

int main(int argc, const char * argv[]) {

  // Setting up random number generator
  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(0,1);

  init_field();
  init_solved_field();
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
