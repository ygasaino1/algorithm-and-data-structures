#ifndef WEIGHTS_H
#define WEIGHTS_H

class Weights{
    private:
        int MAX_WEIGHTS;

    public:
        double* values;
        int number_weights;

        Weights();
        void init(int number_predictor, int random_init);
        ~Weights() = default;
        void update(class Dataset data, double *y_pred, double learning_rate) const;
};
#endif // WEIGHTS_H
