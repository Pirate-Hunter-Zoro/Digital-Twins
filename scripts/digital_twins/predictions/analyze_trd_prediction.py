from scripts.digital_twins.predictions.trd_prediction_analysis import run_trd_prediction_analysis
from scripts.digital_twins.predictions.trd_ranking_analysis import run_trd_ranking_analysis
from scripts.digital_twins.predictions.trd_binning_analysis import run_trd_bin_analysis
from scripts.digital_twins.predictions.trd_sanity_checks import run_trd_sanity_checks
    
if __name__=="__main__":
    run_trd_prediction_analysis()
    run_trd_ranking_analysis()
    run_trd_bin_analysis()
    run_trd_sanity_checks()