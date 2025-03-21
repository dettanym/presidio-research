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

        self.plot_roc(df, self.dataset_name, True)
        self.plot_roc(df, self.dataset_name, False)

        for threshold in self.plotter_per_threshold.keys():
            if threshold == 0.45:
                self.conf_matrix_error_analysis(threshold)


    def conf_matrix_error_analysis(self, threshold: float):
        # Get plotter and results for this threshold
        plotter = self.plotter_per_threshold[threshold]
        results = self.results_per_threshold[threshold]

        # Get confusion matrix
        entities, confmatrix = results.to_confusion_matrix()

        # Log experiment, confusion matrix
        entities_mapping = PresidioAnalyzerWrapper.presidio_entities_map
        self.log_experiment(results, self.dataset_name, entities_mapping, entities, confmatrix)
        masked_or_random = self.dataset_name.split(".")[0].split("_")[3]
        Plotter.plot_confusion_matrix(entities=entities, confmatrix=confmatrix, suffix=masked_or_random)

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

    @staticmethod
    def plot_roc(df:pd.DataFrame, dataset_name: str, is_log: bool):
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
                if PlotterForMultipleEvaluations.should_annotate(pii_type, threshold, is_log, fpr):
                    offset_for_this_point = PlotterForMultipleEvaluations.get_label_position(masked_or_random, pii_type, threshold, is_log)
                    ax.annotate("  %s" % threshold, xy=(fpr, recall), xytext=offset_for_this_point,
                                textcoords="offset points", color=color)
            colorcount += 1

        ax.set_ylabel("True Positive Rate")
        ax.set_xlabel("False Positive Rate")
        if is_log:
            ax.set_xlim(0.00001, 0.02)
            ax.set_xscale("log")
            ax.set_yscale("log")
            is_log_str = "_log_axes"
        else:
            ax.set_xlim(-0.00015, 0.002)
            ax.axline((0, 0), slope=1, linestyle="--", label="Random classifier", color="gray")
            is_log_str = "_normal_axes"
        title_keyword = "representative" if masked_or_random == "masked" else masked_or_random
        plt.suptitle("Receiver operating characteristic", fontsize=16)
        ax.set_title("Dataset with " + title_keyword + " keys")
        fig.legend(title="PII types", loc="outside lower center", ncols=4)
        plt.savefig('plots/roc_' + masked_or_random + is_log_str + '.png')

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
    def get_label_position(masked_or_random: str, pii_type:str, threshold:float, is_log: bool) -> Tuple[int, int]:
        if not is_log:
            length_of_single_digit = 18
            label_height = 10
            default_double_digit_xoffset = -36
            default_single_digit_xoffset = -30
            default_top_height_yoffset = 3

            if pii_type == "AGE":
                return {
                    0.9: (default_single_digit_xoffset, -10)
                }.get(threshold)
            if pii_type == "CREDIT_CARD":
                return {
                    0.9: (default_single_digit_xoffset, -5),
                }.get(threshold)
            if pii_type == "DATE_TIME":
                return {
                    0.9: (default_single_digit_xoffset, -2)
                }.get(threshold)
            if pii_type == "EMAIL_ADDRESS":
                return {
                    0.9: (default_single_digit_xoffset, default_top_height_yoffset - 1 * label_height)
                }.get(threshold)
            if pii_type == "IBAN_CODE":
                return {
                    0.9: (default_single_digit_xoffset, default_top_height_yoffset - 2 * label_height)
                }.get(threshold)
            if pii_type == "IP_ADDRESS":
                return {
                    0.3: (default_single_digit_xoffset, default_top_height_yoffset - 3 * label_height),
                    0.9: (default_single_digit_xoffset, -5)
                        if masked_or_random == "random"
                        else (default_single_digit_xoffset, default_top_height_yoffset - 3 * label_height),
                }.get(threshold)
            if pii_type == "LOCATION":
                return {
                    0.9: (default_single_digit_xoffset + length_of_single_digit, -10),
                }.get(threshold)
            if pii_type == "NRP":
                return {
                    0.9: (default_single_digit_xoffset + 2 * length_of_single_digit, -10)
                }.get(threshold)
            if pii_type == "PERSON":
                return {
                    0.9: (default_single_digit_xoffset + 4 * length_of_single_digit, -10)
                }.get(threshold)
            if pii_type == "PHONE_NUMBER":
                return {
                    0.75: (default_double_digit_xoffset, -5),
                    0.9: (default_single_digit_xoffset + 6 * length_of_single_digit, -10)
                }.get(threshold)
            if pii_type == "PII":
                return {
                    0.75: (default_double_digit_xoffset, -1),
                    0.9: (default_single_digit_xoffset, -5)
                }.get(threshold)
            if pii_type == "URL":
                return {
                    0.6: (default_single_digit_xoffset, default_top_height_yoffset),
                    0.9: (default_single_digit_xoffset, -9)
                        if masked_or_random == "random"
                        else (default_single_digit_xoffset + 1.75 * length_of_single_digit, -5)
                }.get(threshold)
            if pii_type == "US_DRIVER_LICENSE":
                return {
                    0.9: (default_single_digit_xoffset + 3 * length_of_single_digit, -10)
                }.get(threshold)
            if pii_type == "US_SSN":
                return {
                    0.85: (-12, default_top_height_yoffset),
                    0.9: (default_single_digit_xoffset + 5 * length_of_single_digit, -10),
                }.get(threshold)

            raise ValueError('unhandled PII type')

        default_offsets = (-5, 5)
        labelxoffset, labelyoffset = default_offsets
        if pii_type == "US_DRIVER_LICENSE":
            if threshold == 0.4:
                labelyoffset = -2 * labelyoffset
            if threshold == 0.3:
                labelxoffset = 2 * labelxoffset # Shift it more to the left so it doesn't hit the x boundary
                labelyoffset = 0.5 * labelyoffset
            if threshold == 0.6:
                labelxoffset = 3 * labelxoffset
        if pii_type == "URL":
            if threshold == 0.5:
                labelyoffset = 0.5 * labelyoffset
        if pii_type == "IP_ADDRESS":
            if threshold == 0.6:
                labelyoffset = 0.5 *labelyoffset
        if pii_type == "DATE_TIME":
            if threshold == 0.9:
                labelxoffset = 5 * labelxoffset
                labelyoffset = -1 * labelyoffset
        if pii_type == "PHONE_NUMBER":
            if threshold == 0.4:
                labelxoffset = 4 * labelxoffset
        if pii_type == "PII":
            if threshold == 0.4:
                labelxoffset = 4 * labelxoffset
            if threshold == 0.9:
                labelxoffset = 5 * labelxoffset
                labelyoffset = -2 * labelyoffset
            if threshold == 0.5:
                labelxoffset = 4 * labelxoffset
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
    def should_annotate(pii_type: str, threshold: float, is_log: bool, fpr: float) -> bool:
        if not is_log:
            if fpr != 0:
                return False

            if pii_type == "PII" and threshold == 0.85:
                return False

            return True

        if pii_type == "EMAIL_ADDRESS" or pii_type == "IBAN_CODE" or pii_type == "US_SSN":
            return False
        if threshold == 0.45 or threshold == 0.75 or threshold == 0.85:
            if pii_type == "PII":
                return False
        if threshold == 0.9:
            if pii_type == "IP_ADDRESS":
                return False
        if threshold == 0.3:
            if pii_type == "PERSON" or pii_type == "DATE_TIME" or pii_type == "LOCATION" \
                or pii_type =="NRP" or pii_type == "PHONE_NUMBER" or pii_type == "URL" \
                    or pii_type == "IP_ADDRESS":
                return False
        if threshold == 0.4:
            if pii_type == "URL" or pii_type == "IP_ADDRESS":
                return False
        if threshold == 0.5:
            if pii_type == "IP_ADDRESS" or pii_type == "DATE_TIME" \
                or pii_type == "US_DRIVER_LICENSE" or pii_type == "PHONE_NUMBER":
                return False
        if threshold == 0.6:
            if pii_type == "PHONE_NUMBER":
                return False
        return True


def read_dfs_from_file(masked_or_random: str) -> pd.DataFrame:
    dataframe_log_filename = "statistics_per_entity_" + masked_or_random + "_dataset.csv"
    df = pd.read_csv(dataframe_log_filename)
    return df


def print_comparison_between_two_datasets(random_data: pd.DataFrame, masked_data: pd.DataFrame):
    """
    this approach assumes that the disjoint union of these two dataframes
    (with respect to entity type and threshold) will return an empty set.
    """

    random_data = random_data.sort_values(by=["entity", "threshold"], ascending=[True, True]).copy()
    masked_data = masked_data.sort_values(by=["entity", "threshold"], ascending=[True, True]).copy()

    merged = pd.merge(
        random_data,
        masked_data,
        suffixes=('_random', '_masked'),
        how='inner',
        on=["entity", "threshold"],
    )
    for index, row in merged.sort_values(by=["entity", "threshold"], ascending=[True, True]).iterrows():
        if row['recall_random'] < row['recall_masked']:
            print(
                f"GOOD SCENARIO: random recall is LT masked recall for \"{row['entity']}\"@{row['threshold']}")
        if row['recall_random'] == row['recall_masked']:
            print(
                f"GOOD SCENARIO: random recall is EQ masked recall for \"{row['entity']}\"@{row['threshold']}")
        if row['fpr_random'] > row['fpr_masked']:
            print(
                f"GOOD SCENARIO: random FPR is GT masked FPR for \"{row['entity']}\"@{row['threshold']}")
            continue
        if row['fpr_random'] == row['fpr_masked']:
            print(
                f"GOOD SCENARIO: random FPR is EQ masked FPR for \"{row['entity']}\"@{row['threshold']}")
            continue
        else:
            print(
                f"BAD SCENARIO for \"{row['entity']}\"@{row['threshold']}" +
                "\n".join(
                    filter(
                        lambda line: line is not None,
                        [
                            None if row['recall_random'] <= row['recall_masked']
                            else f"    random recall is higher than masked, difference is {row['recall_random'] - row['recall_masked']}",
                            None if row['fpr_random'] >= row['fpr_masked']
                            else f"    random fpr is lower than masked, difference is {row['fpr_masked'] - row['fpr_random']}",
                        ]
                    )
                )
            )



