#include "utils.h"

#include <fstream>
#include <sstream>
#include <string>


// Misc Helper function 
Dataset read_csv(const char* filename)
{
    // Variable Initialization
    double** X;
    double* y;
    int length = 0;
    int number_predictor = 0;

    // Reading File to get the number of x and y data points
    std::ifstream infile{filename};
    std::string line;
    while (std::getline(infile, line))
    {
        length++;
        // Calculate the number of predictors
        if (length == 1)
        {
            int i = 0;
            while (line[i] != '\0')
            {
                if (line[i] == ',')
                {
                    number_predictor++;
                }
                i++;
            }
        }
    }
    infile.close();

    // M-allocating space for X and y
    X = static_cast<double**>(malloc(sizeof(double*) * length));
    for (int i = 0; i < length; i++)
    {
        X[i] = static_cast<double*>(malloc(sizeof(double) * number_predictor));
    }
    y = static_cast<double*>(malloc(sizeof(double) * length));

    // Rereading the file to extract x and y values
    std::ifstream samefile(filename);
    int current_index = 0;
    while (std::getline(samefile, line))
    {
        std::stringstream line_stream{line};
        int current_predictor = 0;
        double number;
        while (line_stream >> number)
        {
            if (current_predictor == number_predictor)
            {
                y[current_index] = number;
            }
            else
            {
                X[current_index][current_predictor] = number;
                current_predictor++;
            }

            if (line_stream.peek() == ',')
            {
                line_stream.ignore();
            }
        }
        current_index++;
    }
    samefile.close();

    auto data = Dataset(X, y, length, number_predictor);
    return data;
}

// int make_csv(const char* filename, float* weights, int number_weights, int number_simulation){
//     return 0;
// }

// Stats Helper function
double mean(const double* y, const int length)
{
    double total = 0;
    for (int i = 0; i < length; i++)
    {
        total = total + y[i];
    }
    return (total / static_cast<double>(length));
}

double sum_residual(const Dataset& data, const double* y_pred, const int current_predictor)
{
    double total = 0;
    for (int i = 0; i < data.length; i++)
    {
        const double residual = (y_pred[i] - data.y[i]);
        total = total + residual * data.x[i][current_predictor];
    }
    return total;
}

double sum_of_square(const double* y, const int length)
{
    // Not the most efficient way of calculating variance, see : https://www.sciencebuddies.org/science-fair-projects/science-fair/variance-and-standard-deviation 
    double total = 0;
    const double y_mean = mean(y, length);
    for (int i = 0; i < length; i++)
    {
        const double residual = (y[i] - y_mean);
        total = total + (residual * residual);
    }
    return total;
}

double residual_sum_of_square(const double* y_pred, const double* y_true, const int length)
{
    double total = 0;
    for (int i = 0; i < length; i++)
    {
        const double residual = (y_true[i] - y_pred[i]);
        total = total + (residual * residual);
    }
    return total;
}

int calculate_r2(const double* y_pred, const double* y_true, const int length)
{
    // Taken from: https://en.wikipedia.org/wiki/Coefficient_of_determination
    const double sum_squared_residual = residual_sum_of_square(y_pred, y_true, length);
    const double sum_squared_total = sum_of_square(y_true, length);
    return static_cast<int>(1 - ((sum_squared_residual / sum_squared_total)));
}

double mean_squared_error(const double* y_pred, const double* y_true, const int length)
{
    return residual_sum_of_square(y_pred, y_true, length) / static_cast<double>(length);
}

double intercept_sum(const double* y_pred, const double* y_true, const int length)
{
    double total = 0;
    for (int i = 0; i < length; i++)
    {
        const double residual = (y_pred[i] - y_true[i]);
        total = total + residual;
    }
    return total;
}

double slope_sum(const double* x, const double* y_pred, const double* y_true, const int length)
{
    double total = 0;
    for (int i = 0; i < length; i++)
    {
        const double residual = (y_pred[i] - y_true[i]);
        total = total + residual * x[i];
    }
    return total;
}
