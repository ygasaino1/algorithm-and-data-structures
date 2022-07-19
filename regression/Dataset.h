#ifndef DATASET_H
#define DATASET_H

class Dataset{
    public:
        double **x{};
        double *y{};
        int number_predictor{};
        int length{};
        double* y_train_{};

        Dataset();
        Dataset(double **x_train,double *y_train, int length_train, int number_predictor_train);
        void copy(const Dataset &data);
        ~Dataset();

        void print_dataset() const;
};
#endif // DATASET_H
