#include "Dataset.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>


Dataset::Dataset() = default;

Dataset::Dataset(double** x_train, double* y_train, int length_train, int number_predictor_train)
{
    x = static_cast<double**>(malloc(sizeof(double*) * length_train));
    for (int i = 0; i < length_train; i++)
    {
        x[i] = static_cast<double*>(malloc(sizeof(double) * number_predictor_train));
        std::memcpy(x[i], x_train[i], sizeof(double) * number_predictor_train);
    }

    y = static_cast<double*>(malloc(sizeof(double) * length_train));
    std::memcpy(y, y_train, sizeof(double) * length_train);

    length = length_train;
    number_predictor = number_predictor_train;
}

void Dataset::copy(const Dataset& data)
{
    x = static_cast<double**>(malloc(sizeof(double*) * data.length));
    for (int i = 0; i < data.length; i++)
    {
        x[i] = static_cast<double*>(malloc(sizeof(double) * data.number_predictor));
        std::memcpy(x[i], data.x[i], sizeof(double) * data.number_predictor);
    }

    y = static_cast<double*>(malloc(sizeof(double) * data.length));
    std::memcpy(y, data.y, sizeof(double) * data.length);

    length = data.length;
    number_predictor = data.number_predictor;
}

Dataset::~Dataset() = default;

void Dataset::print_dataset() const
{
    for (int i = 0; i < length; i++)
    {
        printf("row = %d: \n", i);
        for (int j = 0; j < number_predictor; j++)
        {
            printf("X%d = %f\n", j, x[i][j]);
        }
        printf("Y = %f\n", y[i]);
    }
}
