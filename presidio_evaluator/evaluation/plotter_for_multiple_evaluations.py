import json
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import kaleido as ks
import pandas as pd

from presidio_evaluator.evaluation import EvaluationResult, Plotter, ModelError
from presidio_evaluator.experiment_tracking import get_experiment_tracker, ExperimentTracker
from presidio_evaluator.models import PresidioAnalyzerWrapper


class PlotterForMultipleEvaluations:
    dataset_name: str
    results_per_threshold:  dict[float, EvaluationResult] = {}
    plotter_per_threshold: dict[float, Plotter] = {}
    dataframes: pd.DataFrame

    def __init__(self, dataset_name: str, results_per_threshold: dict[float, EvaluationResult]):
        self.results_per_threshold = results_per_threshold
        self._initialize_plotters()
        self.dataset_name = dataset_name

    def _initialize_plotters(self):
        for threshold, results in self.results_per_threshold.items():
            plotter = Plotter(results=results,
                              output_folder="plots",
                              model_name="Presidio Analyzer",
                              save_as="png",
                              beta=2)
            self.plotter_per_threshold[threshold] = plotter

    def process_multiple_evaluators(self):
        result_dicts = []
        # Get the plotter object, run get_scores, set scores["threshold"] = threshold, get pd.DataFrame(scores)
        for threshold, plotter in self.plotter_per_threshold.items():
            plotter = self.plotter_per_threshold[threshold]
            scores = plotter.get_scores()
            scores["threshold"] = threshold
            df = pd.DataFrame(scores)
            result_dicts.append(df)

        df = pd.concat(result_dicts)
        self.dataframes = df
        self.write_dfs_to_file(df=self.dataframes, dataset_name=self.dataset_name)

        self.plot_roc(df, self.dataset_name)

        # for threshold in self.plotter_per_threshold.keys():
        #     if threshold == 0.4 or threshold == 0.85:
        #         self.conf_matrix_error_analysis(threshold)


    def conf_matrix_error_analysis(self, threshold: float):
        # Get plotter and results for this threshold
        plotter = self.plotter_per_threshold[threshold]
        results = self.results_per_threshold[threshold]

        # Get confusion matrix
        entities, confmatrix = results.to_confusion_matrix()

        # Log experiment, confusion matrix
        entities_mapping = PresidioAnalyzerWrapper.presidio_entities_map
        self.log_experiment(results, self.dataset_name, entities_mapping, entities, confmatrix)
        Plotter.plot_confusion_matrix(entities=entities, confmatrix=confmatrix)

        # Error analysis
        plotter.plot_most_common_tokens()

        # ModelError.most_common_fp_tokens(results.model_errors)
        # fps_df = ModelError.get_fps_dataframe(results.model_errors, entity="PERSON")
        # fps_df[["full_text", "token", "annotation", "prediction"]].head(20)
        # ModelError.most_common_fn_tokens(results.model_errors, n=15)
        # fns_df = ModelError.get_fns_dataframe(results.model_errors, entity="PHONE_NUMBER")
        # fns_df[["full_text", "token", "annotation", "prediction"]].head(20)

    def write_dfs_to_file(self, df: pd.DataFrame, dataset_name: str):
        masked_or_random = dataset_name.split(".")[0].split("_")[3]
        dataframe_log_filename = "statistics_per_entity_" + masked_or_random + "_dataset.csv"
        df = df.sort_values(by=["entity", "threshold"], ascending=[True, True])
        df.to_csv(dataframe_log_filename, index=False)

    def read_dfs_from_file(self, masked_or_random: str) -> pd.DataFrame:
        dataframe_log_filename = "statistics_per_entity_" + masked_or_random + "_dataset.csv"
        df = pd.DataFrame.from_csv(dataframe_log_filename)
        return df

    @staticmethod
    def plot_roc(df:pd.DataFrame, dataset_name: str):
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        df = df[df[
                    "count"] != 0]  # TODO: what does count mean? num of annotations (ground truth) or predictions (observed occurrences)
        # df = df[df["entity"] == "PII"]
        masked_or_random = dataset_name.split(".")[0].split("_")[3]

        colors = plt.get_cmap('tab20').colors
        colorcount = 0
        for pii_type, grp in df.groupby("entity"):
            grp = grp.drop_duplicates(subset=["precision", "recall", "fpr"], keep="last")
            print(pii_type)
            print(grp)
            if pii_type == "PII":
                marker = "*"
                color = "red"
            elif pii_type == "US_SSN":
                color = "black"
            else:
                color = colors[PlotterForMultipleEvaluations.map_pii_type_to_color_index(pii_type)]
                if pii_type == "EMAIL_ADDRESS":
                    marker = [[-1, -1], [-1, 1], [1, 1], [-1, -1]]
                elif pii_type == "IBAN_CODE":
                    marker = [[-1, -1], [1, -1], [1, 1], [-1, -1]]
                elif pii_type == "IP_ADDRESS":
                    marker = "x"
                else:
                    marker = "."
            ax.plot(grp["fpr"], grp["recall"], data=grp, label=pii_type, marker=marker, color=color)
            for (fpr, recall, threshold) in zip(grp["fpr"], grp["recall"], grp["threshold"]):
                if PlotterForMultipleEvaluations.should_annotate(pii_type, threshold):
                    offset_for_this_point = PlotterForMultipleEvaluations.get_label_position(masked_or_random, pii_type, threshold)
                    ax.annotate("  %s" % threshold, xy=(fpr, recall), xytext=offset_for_this_point,
                                textcoords="offset points", color=color)
            colorcount += 1

        ax.axline((0, 0), slope=1, linestyle="--", label="Random classifier", color="gray")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlabel("False Positive Rate")
        # ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0.00001, 0.02)
        ax.set_xscale("log")
        ax.set_xscale("log")
        ax.set_yscale("log")
        title_keyword = "representative" if masked_or_random == "masked" else masked_or_random
        plt.suptitle("Receiver operating characteristic", fontsize=16)
        ax.set_title("Dataset with " + title_keyword + " keys")
        fig.legend(title="PII types", loc="outside lower center", ncols=3)
        plt.savefig('plots/roc_' + masked_or_random + '.png')

    @staticmethod
    def log_experiment(results: EvaluationResult, dataset_name: str, entities_mapping: str,
                       entities: List[str], confmatrix: List[List[int]]):
        # # Set up the experiment tracker to log the experiment for reproducibility
        experiment = get_experiment_tracker()

        # Track model and dataset params
        params = {"dataset_name": dataset_name,
                  "model_name": "Presidio Analyzer"}

        # Log experiment params
        experiment.log_parameters(params)
        experiment.log_parameter("entity_mappings", json.dumps(entities_mapping))

        # Track experiment results
        experiment.log_metrics(results.to_log())

        # Plot and log confusion matrices
        experiment.log_confusion_matrix(matrix=confmatrix,
                                        labels=entities)

        # end experiment
        experiment.end()

    @staticmethod
    def get_label_position(masked_or_random: str, pii_type:str, threshold:float) -> Tuple[int, int]:
        default_offsets = (-5, 5)
        labelxoffset, labelyoffset = default_offsets
        if masked_or_random == "masked":
            return default_offsets
        if pii_type == "US_DRIVER_LICENSE":
            if threshold == 0.4 or threshold == 0.6:
                labelyoffset = -2 * labelyoffset
            if threshold == 0.3:
                labelxoffset = 2 * labelxoffset # Shift it more to the left so it doesn't hit the x boundary
                labelyoffset = 0.5 * labelyoffset
        if pii_type == "URL":
            if threshold == 0.5:
                labelyoffset = 0.5 * labelyoffset
        if pii_type == "IP_ADDRESS":
            if threshold == 0.6:
                labelyoffset = 0.5 *labelyoffset
        if pii_type == "DATE_TIME":
            if threshold == 0.4:
                labelyoffset = -2 * labelyoffset
            if threshold == 0.6:
                labelxoffset = 5 * labelxoffset
            if threshold == 0.9:
                labelxoffset = 5 * labelxoffset
                labelyoffset = -1 * labelyoffset
        if pii_type == "PHONE_NUMBER":
            if threshold == 0.4:
                labelxoffset = 4 * labelxoffset
            if threshold == 0.75:
                labelyoffset = -2 * labelyoffset
                labelxoffset = 4 * labelxoffset
        if pii_type == "PII":
            if threshold == 0.4:
                labelyoffset = -2 * labelyoffset
            if threshold == 0.9:
                labelxoffset = 5 * labelxoffset
                labelyoffset = -2 * labelyoffset
            if threshold == 0.5:
                labelxoffset = 5 * labelxoffset
            if threshold == 0.6:
                labelxoffset = 4 * labelxoffset
        return labelxoffset, labelyoffset

    # Maps each pii type to a color index in any paired color scheme with at least 20 colors
    # The entity type "PII" should separately be set to black outside of this function
    @staticmethod
    def map_pii_type_to_color_index(pii_type: str) -> int:
        mapping = {
            #First set light colors to all lines
            "URL": 10,
            "DATE_TIME": 3,
            "PHONE_NUMBER": 5,
            "US_DRIVER_LICENSE": 9,
            # Rest are the same
            "CREDIT_CARD": 1,
            "PERSON": 8,
            "NRP": 7,
            "LOCATION": 12,
            # Out of the rest, the next two show up as points on top of other lines
            "IP_ADDRESS":0,
            # These three don't show up at all
            "EMAIL_ADDRESS": 4,
            "IBAN_CODE": 2,
            "AGE": 11
        }
        return mapping[pii_type]

    @staticmethod
    def should_annotate(pii_type: str, threshold: float) -> bool:
        if pii_type == "PII" and (threshold == 0.45 or threshold == 0.75 or threshold == 0.85):
            return False
        else:
            if pii_type == "EMAIL_ADDRESS" or pii_type == "IBAN_CODE" or pii_type == "US_SSN":
                return False
            if pii_type == "IP_ADDRESS" and threshold == 0.9:
                return False
            if pii_type == "URL" and threshold == 0.6:
                return False
            return True
