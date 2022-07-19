#ifndef UTILS_H
#define UTILS_H

#include "Dataset.h"

Dataset read_csv(const char* filename);
// int make_csv(const char* filename, Weights weights);
double mean(const double* y, const int length);
double sum_of_square(const double* y, const int length);
double sum_residual(const Dataset& data, const double* y_pred, const int current_predictor);
double residual_sum_of_square(const double* y_pred, const double* y_true, const int length);
int calculate_r2(const double* y_pred, const double* y_true, const int length);
double mean_squared_error(const double* y_pred, const double* y_true, const int length);

double intercept_sum(const double* y_pred, const double* y_true, int length);
double slope_sum(const double* x, const double* y_pred, const double* y_true, int length);

#endif
