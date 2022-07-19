#include <cstdio>
#include <iostream>
#include <cstring>

#include "utils.h"
#include "Weights.h"


// Model class for MSE Linear Regression
// Update: https://en.wikipedia.org/wiki/Linear_regression
class LinearRegressionModel
{
    // Models Variable
    Dataset data;
    Weights weights;

    // Public function for user
public:
    // Constructor
    LinearRegressionModel(const Dataset& data_train)
    {
        // Setting Variables
        data.copy(data_train);
        weights.init(data.number_predictor, 0);
    }

    void print_weights() const
    {
        char function_string[1000];
        printf("Number weights = %d\n", weights.number_weights);
        strcpy_s(function_string, "y = ");
        for (int i = 0; i < weights.number_weights; i++)
        {
            printf("Weights %d is = %f\n", i, weights.values[i]);
            char weight[20];
            sprintf_s(weight, "%.2f * x%d", weights.values[i], i);
            strcat_s(function_string, weight);
            if (i == weights.number_weights - 1)
            {
                strcat_s(function_string, "\n");
            }
            else
            {
                strcat_s(function_string, " + ");
            }
        }
        printf("%s\n", function_string);
    }

    // Train the regression model with some data
    void train(int max_iteration, const double learning_rate) const
    {
        // M-allocating some space for prediction
        const auto y_pred = static_cast<double*>(std::malloc(sizeof(double) * data.length));

        while (max_iteration > 0)
        {
            fit(y_pred);
            weights.update(data, y_pred, learning_rate);

            const double mse = mean_squared_error(y_pred, data.y, data.length);

            if (max_iteration % 100 == 0)
            {
                print_weights();
                std::cout << "Iteration left: " << max_iteration << "; MSE = " << mse << "\n";
            }
            max_iteration--;
        }
        free(y_pred);
    }

    double predict(const double* x) const
    {
        double prediction = 0;
        for (int i = 0; i < weights.number_weights; i++)
        {
            prediction = prediction + weights.values[i] * x[i];
        }
        return prediction;
    }

    // Private function for algorithm
private:
    // fit a line given some x and weights
    void fit(double* y_pred) const
    {
        for (int i = 0; i < data.length; i++)
        {
            y_pred[i] = predict(data.x[i]);
        }
    }
};


int main()
{
    // Variable Initialization
    const auto filename = "test.csv";
    std::cout << "Reading CSV \n";
    const Dataset data = read_csv(filename);


    // Regression Variables
    constexpr int max_iteration = 1000;
    constexpr double learning_rate = 0.1;

    // Training
    std::cout << "Making LinearRegressionModel \n";
    const auto linear_reg = LinearRegressionModel(data);
    std::cout << "Training \n";
    linear_reg.train(max_iteration, learning_rate);

    std::cout << "Testing \n";
    // Testing
    double x_test[2];
    x_test[0] = 1;
    x_test[1] = 123;
    const double y_test = linear_reg.predict(x_test);
    linear_reg.print_weights();
    std::cout << "Testing for X0 = " << x_test[0] << ", X1 = " << x_test[1] << "\n";
    std::cout << "y = " << y_test << "\n";
}
