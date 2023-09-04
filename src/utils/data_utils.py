import numpy
import keras.backend as K

def standardise_sample(sample):

    return (sample - sample.min()) / ((sample.max() - sample.min()) + K.epsilon())


def normalise_sample(sample, mean, stddev):

    sample -= mean
    sample /= stddev + K.epsilon()

    return sample
