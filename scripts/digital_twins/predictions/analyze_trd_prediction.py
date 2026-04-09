from scripts.digital_twins.predictions.trd_prediction_computation import run_trd_prediction_computation
from scripts.digital_twins.predictions.trd_ranking_analysis import run_trd_ranking_analysis
from scripts.digital_twins.predictions.trd_binning_analysis import run_trd_bin_analysis
from scripts.digital_twins.predictions.trd_sanity_checks import run_trd_sanity_checks

from scripts.shared.utils import VectorSource
 
if __name__=="__main__":
    for source in VectorSource:
        run_trd_prediction_computation(source)
        run_trd_ranking_analysis(source)
        run_trd_bin_analysis(source)
        run_trd_sanity_checks(source)