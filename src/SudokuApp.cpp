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

int current_state[9][9];
int tentative_new_state[9][9];
int new_state[9][9];

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

double cost(int state[9][9]) {
  int cost = 0;

  // Iterate over each cell in the field
  for (int y=0; y<9; y++) {
    for (int x=0; x<9; x++) {
      int current_cell = state[y][x];
      // In each row current cell must be unique
      for (int l=0; l<9; l++) if (current_cell == state[y][l] && x!=l) cost++;
      // In each column current cell must be unique
      for (int l=0; l<9; l++) if (current_cell == state[l][x] && y!=l) cost++;
      // In each 3x3-block current cell must be unique
      int m = x/3;
      int n = y/3;
      for (int l=0; l<3; l++) {
        for (int k=0; k<3; k++) {
          if (current_cell == state[3*n+l][3*m+k] && y!=3*n+l && x!=3*m+k) cost++;
        }
      }
    }
  }
  return 2000-cost;
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
      cv::putText(image, std::to_string(current_state[i][j]), cv::Point((j*s)+5,(i*s)+s-6),
          cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,0), 1, CV_AA);

    }
  }
  cv::imshow("field",image);
  cv::waitKey(0);
}

void init_field (int field[9][9]) {
  // Setting up random number generator
  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(1,10);

  // Init field with 9 numbers of each number
  for (int j=0; j<9; j++) {
    for (int i=0; i<9; i++) {
      field[i][j] = i;
    }
  }

  // Randomly swap the numbers
}

void init_solved_field () {
  // Setting up random number generator
  current_state[0][0] = 8;
  current_state[0][1] = 2;
  current_state[0][2] = 7;
  current_state[0][3] = 1;
  current_state[0][4] = 5;
  current_state[0][5] = 4;
  current_state[0][6] = 3;
  current_state[0][7] = 9;
  current_state[0][8] = 6;

  current_state[1][0] = 9;
  current_state[1][1] = 6;
  current_state[1][2] = 5;
  current_state[1][3] = 3;
  current_state[1][4] = 2;
  current_state[1][5] = 7;
  current_state[1][6] = 1;
  current_state[1][7] = 4;
  current_state[1][8] = 8;

  current_state[2][0] = 3;
  current_state[2][1] = 4;
  current_state[2][2] = 1;
  current_state[2][3] = 6;
  current_state[2][4] = 8;
  current_state[2][5] = 9;
  current_state[2][6] = 7;
  current_state[2][7] = 5;
  current_state[2][8] = 2;

  current_state[3][0] = 5;
  current_state[3][1] = 9;
  current_state[3][2] = 3;
  current_state[3][3] = 4;
  current_state[3][4] = 6;
  current_state[3][5] = 8;
  current_state[3][6] = 2;
  current_state[3][7] = 7;
  current_state[3][8] = 1;

  current_state[4][0] = 4;
  current_state[4][1] = 7;
  current_state[4][2] = 2;
  current_state[4][3] = 5;
  current_state[4][4] = 1;
  current_state[4][5] = 3;
  current_state[4][6] = 6;
  current_state[4][7] = 8;
  current_state[4][8] = 9;

  current_state[5][0] = 6;
  current_state[5][1] = 1;
  current_state[5][2] = 8;
  current_state[5][3] = 9;
  current_state[5][4] = 7;
  current_state[5][5] = 2;
  current_state[5][6] = 4;
  current_state[5][7] = 3;
  current_state[5][8] = 5;

  current_state[6][0] = 7;
  current_state[6][1] = 8;
  current_state[6][2] = 6;
  current_state[6][3] = 2;
  current_state[6][4] = 3;
  current_state[6][5] = 5;
  current_state[6][6] = 9;
  current_state[6][7] = 1;
  current_state[6][8] = 4;

  current_state[7][0] = 1;
  current_state[7][1] = 5;
  current_state[7][2] = 4;
  current_state[7][3] = 7;
  current_state[7][4] = 9;
  current_state[7][5] = 6;
  current_state[7][6] = 8;
  current_state[7][7] = 2;
  current_state[7][8] = 3;

  current_state[8][0] = 2;
  current_state[8][1] = 3;
  current_state[8][2] = 9;
  current_state[8][3] = 8;
  current_state[8][4] = 4;
  current_state[8][5] = 1;
  current_state[8][6] = 5;
  current_state[8][7] = 6;
  current_state[8][8] = 7;
}

int main(int argc, const char * argv[]) {

  // Setting up random number generator
  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(0,1);

  init_field(current_state);
  std::cout << "Cost: " << cost(current_state) << std::endl;
  show_state();

  int best_cost = 0;

  // Start iterating
  for (unsigned int it=0; it<max_iterations; it++) {
    // Sample tentative new state from propsal density
    init_field(tentative_new_state);
    std::cout << "Tentative Cost: " << cost(tentative_new_state) << std::endl;

    // Compute factor a

    float a = cost(tentative_new_state) / cost(current_state); // Compute acceptance rate
    std::cout << "a=" << a << std::endl;
    //Decide wether to accept tentative new state
    if(a >= 1) {
        for (int j=0; j<9; j++) {
          for (int i=0; i<9; i++) {
            new_state[i][j] = tentative_new_state[i][j];
          }
        }
    } else { // Accept new state with probability a
      if (a >= uniform_density(generator)){
          for (int j=0; j<9; j++) {
            for (int i=0; i<9; i++) {
              new_state[i][j] = tentative_new_state[i][j];
            }
          }
      } else {
          for (int j=0; j<9; j++) {
            for (int i=0; i<9; i++) {
              new_state[i][j] = current_state[i][j];
            }
          }
      }
    }

  // Prepare for next round...
    for (int j=0; j<9; j++) {
      for (int i=0; i<9; i++) {
        current_state[i][j] = new_state[i][j];
      }
    }

    int current_cost = cost(current_state);
    if (current_cost > best_cost) best_cost = current_cost;

    std::cout << "Cost: " << current_cost << " Best=" << best_cost << std::endl;
    show_state();
    std::cout << "---------------------------------------------" << std::endl;


  }

  /*    std::normal_distribution<float> proposal_density(current_state,3); // Normal distribution at current state, mean=1
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
  }*/
  std::cout << "DONE" << std::endl;
  return 0;
}
