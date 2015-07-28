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
int current_cost[9][9];

int tentative_new_state[9][9];
int tentative_new_cost[9][9];

int new_state[9][9];
int new_cost[9][9];

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

int compute_cost(int state[9][9], int cost[9][9]) {
  // Iterate over each cell in the field
  for (int y=0; y<9; y++) {
    for (int x=0; x<9; x++) {
      // First set cost to zero
      cost[y][x] = 0;
      int current_cell = state[y][x];
      // Must appear exactly once in a row
      for (int l=0; l<9; l++) if (current_cell == state[y][l] && x!=l) cost[y][x]++;
      // Must appear exactly once in a column
      for (int l=0; l<9; l++) if (current_cell == state[l][x] && y!=l) cost[y][x]++;
      // Must appear exactly once in each of the 3-by-3 subgrids
      int n = y/3;
      int m = x/3;
      for (int l=0; l<3; l++) {
        for (int k=0; k<3; k++) {
          //std::cout << "Grid " << n << "x" << m << " : " << x << "-" << y << std::endl;
          //std::cout << "\t" << y << "x" << x << " - " << 3*n+l << "x" <<  3*m+k << std::endl;
          if (current_cell == state[3*n+l][3*m+k] && !(y==3*n+l && x==3*m+k)) {
            cost[y][x]++;
          }
        }
      }
    }
  }

  // Accumulate costs of all cells
  int all_cost = 0;
  for (int y=0; y<9; y++) {
    for (int x=0; x<9; x++) {
        all_cost += cost[y][x];
    }
  }
  return all_cost;
}

void show_state(int state[9][9], int cost[9][9]) {
  double m = 0; // outer margin size
  double s = 35; // width of cell
  cv::Mat image(s*9+2*m,s*9+2*m,CV_32FC3,cv::Scalar(255,255,255));
  cv::rectangle(image, cv::Point(m,m), cv::Point(s*9+m,s*9+m), cv::Scalar(0,0,0), 2);
  for (int j=0; j<10; j++) {
    for (int i=0; i<10; i++) {
      if (i%3==0 && j%3==0) {
        cv::rectangle(image, cv::Point((j*s),(i*s)), cv::Point(s*3,s*3), cv::Scalar(0,0,0), 2);
      }
      cv::rectangle(image, cv::Point((j*s),(i*s)), cv::Point(s,s), cv::Scalar(0,0,0), 1);
      cv::putText(image, std::to_string(state[i][j]), cv::Point((j*s)+10,(i*s)+s-11),
          cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,0), 1, CV_AA);
      cv::Scalar color = (cost[i][j]==0)?cv::Scalar(0,50,0) : cv::Scalar(0,0,255);
      cv::putText(image, std::to_string(cost[i][j]), cv::Point((j*s)+1,(i*s)+s-4),
          cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1, CV_AA);
    }
  }
  cv::imshow("Field",image);
  cv::waitKey(1);
}

/**
 * @brief init_field - fills the field with numbers at random positions and guarantees
 *                     that each number occurs exactly 9 times.
 * @param field - the field to be initialized
 */
void init_field (int field[9][9]) {

  // Init field with 9 instances of each number
  for (int j=0; j<9; j++) for (int i=0; i<9; i++) field[j][i] = i+1;

  // Setting up random number generator
  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(0,9);

  // Randomly swap the numbers
  for (int i=0; i<81; i++) {
      int x1 = uniform_density(generator);
      int y1 = uniform_density(generator);
      int x2 = uniform_density(generator);
      int y2 = uniform_density(generator);
      //std::cout << "Swapping " << y1 << "x" << x1 << " with " << y2 << "x" << x2 << std::endl;
      int tmp = field[y2][x2];
      field[y2][x2] = field[y1][x1];
      field[y1][x1] = tmp;
  }
}

/**
 * @brief pick_sample
 * @param cost
 * @return
 */
std::pair<int, int> pick_sample(int cost[9][9]) {

  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(0,1);

  double F[9*9]; // Cumulative distribution function
  int index = 0;
  F[index] = std::exp(cost[0][0]);
  for (int y=0; y<9; y++) for (int x=0; x<9; x++) {
    if (index==0) { F[index] = std::exp(cost[0][0]); index++; continue; }
    F[index] = F[index-1] + std::exp(cost[y][x]);
    index++;
  }
  // Normalize
  unsigned int Z = 0; // Normalization factor
  for (int y=0; y<9; y++) for (int x=0; x<9; x++) Z+=std::exp(cost[y][x]);
  for (int i=0; i<81; i++) F[i]/=Z;

  float rnd = uniform_density(generator);
  int sample = 0;
  while (rnd > F[sample]) sample++;

  std::pair<int, int> s;
  s.first = sample/9;
  s.second = sample-9*s.first;
  return s;
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
  compute_cost(current_state, current_cost);
  show_state(current_state, current_cost);
  cv::waitKey(0);

  int best_cost = 0;

  // Start iterating
  for (unsigned int it=0; it<max_iterations; it++) {

    // Sample tentative new state
    for (int j=0; j<9; j++) for (int i=0; i<9; i++) tentative_new_state[j][i] = current_state[j][i];

    std::pair<int, int> s1 = pick_sample(current_cost);
    std::pair<int, int> s2 = pick_sample(current_cost);

    std::cout << "s1=" << s1.first << "x" << s1.second << std::endl;
    std::cout << "s2=" << s2.first << "x" << s2.second << std::endl;

    // Swap two cells
    tentative_new_state[s1.first][s1.second] = current_state[s2.first][s2.second];
    tentative_new_state[s2.first][s2.second] = current_state[s1.first][s1.second];

    std::cout << "current_cost=" << compute_cost(current_state, current_cost) << std::endl;
    std::cout << "tentative_cost=" << compute_cost(tentative_new_state, tentative_new_cost) << std::endl;

    // Compute factor a
    double temperature = 1.0;
    double a = std::exp(( (double)compute_cost(current_state, current_cost) -
                         (double)compute_cost(tentative_new_state, tentative_new_cost)
                         ) / temperature);
    std::cout << "a=" << a << std::endl;

    //Decide wether to accept tentative new state
    if (a >= 1) { // Accept new state
        for (int j=0; j<9; j++) for (int i=0; i<9; i++) new_state[j][i] = tentative_new_state[j][i];
    } else { // Accept new state with probability a
      if (a >= uniform_density(generator)) { // Accept new state
          for (int j=0; j<9; j++) for (int i=0; i<9; i++) new_state[j][i] = tentative_new_state[j][i];
      } else { // Reject new state, keep old/current
          for (int j=0; j<9; j++) for (int i=0; i<9; i++) new_state[j][i] = current_state[j][i];
      }
    }

    // Prepare for next round...
    for (int j=0; j<9; j++) for (int i=0; i<9; i++) current_state[j][i] = new_state[j][i];

    int current_cost_all = compute_cost(current_state, current_cost);
    if (current_cost_all > best_cost) best_cost = current_cost_all;

    std::cout << "Cost: " << current_cost_all << " Best=" << best_cost << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    show_state(current_state, current_cost);
    if (current_cost_all==0) break;
  }
  std::cout << "DONE" << std::endl;
  show_state(current_state, current_cost);
  cv::waitKey(0);
  return 0;
}
