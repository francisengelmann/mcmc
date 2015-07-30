//
//  Solving Sudoku using MCMC.
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
unsigned int max_iterations = 1000000; // Number of MCMC iterations
unsigned int display_time_ms = 5; // How long to show the field image

// Number of samples, used for normalizing the estimated distribution
unsigned int sample_count = 1;
std::map<int, int> estimtated_distribution;

// States and corresponding costs
int current_state[9][9];
int current_cost[9][9];
int tentative_new_state[9][9];
int tentative_new_cost[9][9];
int new_state[9][9];
int new_cost[9][9];

/**
 * @brief compute_cost - Computes the global cost of the field and the cost of each cell.
 * @param state - State for which to compute the cost.
 * @param cost - Array that will contain the computed cost.
 * @return Overall cost i.e. the sum of the cost of each cell.
 */
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
          if (current_cell == state[3*n+l][3*m+k] && !(y==3*n+l && x==3*m+k)) {
            cost[y][x]++;
          }
        }
      }
    }
  }

  // Accumulate costs of all cells
  int all_cost = 0;
  for (int y=0; y<9; y++) for (int x=0; x<9; x++) all_cost += cost[y][x];
  return all_cost;
}

/**
 * @brief show_state - Displays the current state of the field with the associated costs.
 * @param state - State to be displayed.
 * @param cost - Costs to be displayed in small letters in each cell.
 */
void show_state(int state[9][9], int cost[9][9]) {

  // Visualization parameters of the field.
  double s = 35; // Width of cell.
  cv::Scalar color_correct = cv::Scalar(0,50,0); // Color of text for cell with cost=0.
  cv::Scalar color_not_correct = cv::Scalar(0,0,255); // Color of text for cell with cost>0.

  // Init image
  cv::Mat image(s*9+2,s*9+2,CV_32FC3,cv::Scalar(255,255,255));
  cv::rectangle(image, cv::Point(0,0), cv::Point(s*9,s*9), cv::Scalar(0,0,0), 2);

  // Iterate over each cell
  for (int j=0; j<10; j++) {
    for (int i=0; i<10; i++) {

      // Draw 3x3-subgrid
      if (i%3==0 && j%3==0) cv::rectangle(image, cv::Point((j*s),(i*s)), cv::Point(s*3,s*3), cv::Scalar(0,0,0), 2);
      cv::rectangle(image, cv::Point((j*s),(i*s)), cv::Point(s,s), cv::Scalar(0,0,0), 1);

      // Draw text of cells current label
      cv::putText(image, std::to_string(state[i][j]), cv::Point((j*s)+10,(i*s)+s-11),
          cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0, cv::Scalar(0,0,0), 1, CV_AA);

      // Draw text of cells current cost
      cv::Scalar color = (cost[i][j]==0) ? color_correct : color_not_correct;
      cv::putText(image, std::to_string(cost[i][j]), cv::Point((j*s)+1,(i*s)+s-4),
          cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, color, 1, CV_AA);
    }
  }

  // Display the current field for specified time
  cv::imshow("Field",image);
  cv::waitKey(display_time_ms);
}

/**
 * @brief init_field - Fills the field with numbers at random positions and guarantees
 *                     that each number occurs exactly 9 times.
 * @param field - Field to be initialized.
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
      int tmp = field[y2][x2];
      field[y2][x2] = field[y1][x1];
      field[y1][x1] = tmp;
  }
}

/**
 * @brief pick_sample - Samples a cell, prefering cells with heigth cost.
 * @param cost - Cost of the state.
 * @return Coordinate pair of the sampled cell.
 */
std::pair<int, int> pick_sample(int cost[9][9]) {

  // Compute cumulative distribution function (CDF)
  double F[9*9];
  int index = 0;
  F[index] = std::exp(cost[0][0]);
  for (int y=0; y<9; y++) for (int x=0; x<9; x++) {
    if (index==0) { F[index] = std::exp(cost[0][0]); index++; continue; }
    F[index] = F[index-1] + std::exp(cost[y][x]);
    index++;
  }

  // Normalize CDF
  unsigned int Z = 0; // Normalization factor
  for (int y=0; y<9; y++) for (int x=0; x<9; x++) Z+=std::exp(cost[y][x]);
  for (int i=0; i<81; i++) F[i]/=Z;

  // Initialize random number generator
  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(0,1);

  // Sample from discrete probabilty distribution c.f. [3]
  float rnd = uniform_density(generator);
  int sample = 0;
  while (rnd > F[sample]) sample++;

  // Build 2D-coordinates from 1D-index and return
  std::pair<int, int> s;
  s.first = sample/9;
  s.second = sample-9*s.first;
  return s;
}

int main(int argc, const char * argv[]) {

  // Setting up random number generator.
  std::mt19937 generator(std::random_device{}());
  std::uniform_real_distribution<float> uniform_density(0,1);

  // Inti field and display it.
  init_field(current_state);
  compute_cost(current_state, current_cost);
  show_state(current_state, current_cost);
  cv::waitKey(0);

  // Start iterating.
  for (unsigned int it=0; it<max_iterations; it++) {

    // Sample tentative new state
    for (int j=0; j<9; j++) for (int i=0; i<9; i++) tentative_new_state[j][i] = current_state[j][i];

    std::pair<int, int> s1 = pick_sample(current_cost);
    std::pair<int, int> s2 = pick_sample(current_cost);

    std::cout << "s1=" << s1.first << "x" << s1.second << std::endl;
    std::cout << "s2=" << s2.first << "x" << s2.second << std::endl;

    // Swap two cells.
    tentative_new_state[s1.first][s1.second] = current_state[s2.first][s2.second];
    tentative_new_state[s2.first][s2.second] = current_state[s1.first][s1.second];

    std::cout << "Current_cost=" << compute_cost(current_state, current_cost) << std::endl;
    std::cout << "Tentative_cost=" << compute_cost(tentative_new_state, tentative_new_cost) << std::endl;

    // Compute factor a.
    double temperature = 1.0;
    double a = std::exp(( (double)compute_cost(current_state, current_cost) -
                         (double)compute_cost(tentative_new_state, tentative_new_cost)
                         ) / temperature);
    std::cout << "a=" << a << std::endl;

    //Decide wether to accept tentative new state.
    if (a >= 1) { // Accept new state
        for (int j=0; j<9; j++) for (int i=0; i<9; i++) new_state[j][i] = tentative_new_state[j][i];
    } else { // Accept new state with probability a
      if (a >= uniform_density(generator)) { // Accept new state
          for (int j=0; j<9; j++) for (int i=0; i<9; i++) new_state[j][i] = tentative_new_state[j][i];
      } else { // Reject new state, keep old/current
          for (int j=0; j<9; j++) for (int i=0; i<9; i++) new_state[j][i] = current_state[j][i];
      }
    }

    // Set current state to new state for next round.
    for (int j=0; j<9; j++) for (int i=0; i<9; i++) current_state[j][i] = new_state[j][i];

    // Display current overall cost.
    int current_cost_all = compute_cost(current_state, current_cost);
    std::cout << "Cost: " << current_cost_all << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    show_state(current_state, current_cost);

    // If cost==0 we can stop.
    if (current_cost_all==0) break;
  }
  std::cout << "-._.-=DONE=-._.-" << std::endl;
  show_state(current_state, current_cost);
  cv::waitKey(0);
  return 0;
}