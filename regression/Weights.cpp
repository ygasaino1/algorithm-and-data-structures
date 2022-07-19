#include "Weights.h"

#include <cstdlib>
#include <ctime>

#include "Dataset.h"
#include "utils.h"

#define noSUM

Weights::Weights(): MAX_WEIGHTS(0), values(nullptr), number_weights(0)
{
    //nothing more
};

void Weights::init(int number_predictor, int random_init)
{
    // Random Init Variables
    MAX_WEIGHTS = 100;
    srand(static_cast<unsigned>(time(0))); // random number generator

    number_weights = number_predictor;
    values = static_cast<double*>(std::malloc(sizeof(double) * number_weights));
    for (int i = 0; i < number_weights; i++)
    {
        if (random_init == 1)
        {
            values[i] = (rand() % MAX_WEIGHTS);
        }
        else
        {
            values[i] = 0;
        }
    }
}

void Weights::update(Dataset data, double* y_pred, double learning_rate) const
{
    const double multiplier = learning_rate / static_cast<double>(data.length);
    // Update each weights
    for (int i = 0; i < number_weights; i++)
    {
        const double sum = (sum_residual(data, y_pred, i));
#ifdef SUM
        printf("Sum = %f\n", sum);
#endif
        
        values[i] = values[i] - multiplier * sum;
    }
}
