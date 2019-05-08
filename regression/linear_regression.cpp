#include <iostream>

int main(){
    std::cout << "Result" << "\n";
}

// NOTES:
// Following this tutorial:
// https://towardsdatascience.com/introduction-to-machine-learning-algorithms-linear-regression-14c4e325882a

// Linear Regression:
// y = a_0 + a_1 * x  
// we want to find the best a_0 and a_1 for the dataset

// Cost Function:
// Need a cost-function and want to minimize that (minimization problem now)
// Mean Squared Error = 1/n * (Sum (y_predi - yi)^2 from i = 1 to n)
// MSE is just a type of error we can choose from

// Gradient Descent
// Used to minimize MSE cost function to find best a_0 and a_1
// Big learning rate = big jumps, small learning rate = super small jumps (longer)
// If cost function not convex can be stuck in a local minima (not happening in linear regression)
// Need gradient, so we take partial derivative with respect to a_0 and a_1.
// Calculus is used to find partial derivative

// How to update
// a -> learning rate
// new_a0 = a0 - (2a/n)* (Sum (y_predi - yi) from i = 1 to n)
// new_a1 = a1 - (2a/n)* (Sum (y_predi - yi)*xi from i = 1 to n)

// How to calculate score (R2 score)