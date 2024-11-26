import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_prepared.load import load
from model.bayesian import bayesian_model
from model.lib import msfe,plot_predictions_vs_actual


def main():
    training_dataset, testing_dataset = load()
    
    #bayesian model prediction data
    bayesian_predictions=bayesian_model(training_dataset, testing_dataset)
    #bayesian model visualization(model_prediction VS real_Y)
    
    
    
    # calculate error
    #error_bayesian=msfe(bayesian_predictions, testing_realY)
    
    
    
    # comparing error
    
    
if __name__ == "__main__":
    main()