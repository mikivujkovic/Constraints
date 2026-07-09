import math

import numpy as np
import pandas as pd
from scipy import stats


def calculate_kruskall_wallis(groups):
    """
    Calculate Kruskal-Wallis H test statistic

    Args:
        groups: List of arrays containing the data for each group

    Returns:
        H statistic
    """
    # Combine all values for ranking
    all_values = []
    group_sizes = []

    for group in groups:
        all_values.extend(group)
        group_sizes.append(len(group))

    total_size = len(all_values)

    # Create pairs of (value, original group index)
    indexed_values = []
    for i, group in enumerate(groups):
        for value in group:
            indexed_values.append((value, i))

    # Sort by value
    indexed_values.sort(key=lambda x: x[0])

    # Assign ranks, handling ties
    ranks = [0] * total_size
    rank_sum_by_group = [0] * len(groups)
    group_counts = [0] * len(groups)

    i = 0
    while i < total_size:
        j = i
        # Find tied values
        while j < total_size - 1 and indexed_values[j + 1][0] == indexed_values[i][0]:
            j += 1

        # Calculate average rank for the tie group
        avg_rank = (i + j) / 2 + 1

        # Assign ranks and update rank sums
        for k in range(i, j + 1):
            group_idx = indexed_values[k][1]
            rank_sum_by_group[group_idx] += avg_rank
            group_counts[group_idx] += 1

        i = j + 1

    # Calculate H statistic
    h = 0
    for i in range(len(groups)):
        h += (rank_sum_by_group[i] ** 2) / group_counts[i]

    h = (12 / (total_size * (total_size + 1))) * h - 3 * (total_size + 1)

    return h


def calculate_cliffs_delta(group1, group2):
    """
    Calculate Cliff's delta effect size

    Args:
        group1: First data array
        group2: Second data array

    Returns:
        Cliff's delta value
    """
    dominance = 0

    for x in group1:
        for y in group2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
            # Equal values contribute 0

    return dominance / (len(group1) * len(group2))


def main():
    # Read the CSV file
    df = pd.read_csv("/data/LLM_evaluations.csv")

    # Define dimension categories
    dimensions = {
        "Visual Design": [
            "color_harmony_contrast",
            "typography_readability",
            "spatial_layout_balance",
            "visual_hierarchy",
            "design_consistency",
        ],
        "Content Quality": [
            "message_clarity",
            "brand_voice_consistency",
            "cta_effectiveness",
            "value_proposition",
            "content_structure",
        ],
        "User Experience": [
            "navigation_intuitiveness",
            "information_architecture",
            "mobile_responsiveness",
            "load_time_performance",
            "interaction_design",
        ],
        "Business Effectiveness": [
            "conversion_potential",
            "brand_alignment",
            "target_audience",
            "competitive_differentiation",
            "market_readiness",
        ],
    }

    # List of all dimensions
    all_dimensions = []
    for category, dims in dimensions.items():
        all_dimensions.extend(dims)

    # List of all generation types and LLMs
    generation_types = df["generation_type"].unique().tolist()
    llms = df["id"].unique().tolist()

    # Prepare output text file
    with open("llms_results.txt", "w") as outfile:
        outfile.write("LLMs SURVEY ANALYSIS RESULTS\n")
        outfile.write("====================================\n\n")

        # Section 1: Basic Statistics
        outfile.write("1. BASIC STATISTICS\n")
        outfile.write("===================\n\n")

        outfile.write("1.1 Mean scores for each dimension by generation approach\n")
        outfile.write("---------------------------------------------------------\n\n")

        # Calculate means for each dimension by generation type
        outfile.write(
            f"{'Dimension':<30} {' '.join(generation_types):<45} {'Overall':<10}\n"
        )
        outfile.write(f"{'-' * 30} {'-' * 45} {'-' * 10}\n")

        for category, dims in dimensions.items():
            for dim in dims:
                # Calculate means for each generation type and overall
                means = [
                    df[df["generation_type"] == gen_type][dim].mean()
                    for gen_type in generation_types
                ]
                overall_mean = df[dim].mean()

                # Format the line with generation type means and overall mean
                mean_str = " ".join([f"{mean:.4f}" for mean in means])
                outfile.write(f"{dim:<30} {mean_str:<45} {overall_mean:.4f}\n")

            # Write category means
            cat_means = [
                df[df["generation_type"] == gen_type][dims].values.flatten().mean()
                for gen_type in generation_types
            ]
            overall_cat_mean = df[dims].values.flatten().mean()

            cat_mean_str = " ".join([f"{mean:.4f}" for mean in cat_means])
            outfile.write(
                f"{category + ' (Category)':<30} {cat_mean_str:<45} {overall_cat_mean:.4f}\n"
            )
            outfile.write("\n")

        # Section 2: Standard Deviations
        outfile.write(
            "1.2 Standard deviation for each dimension by generation approach\n"
        )
        outfile.write(
            "--------------------------------------------------------------\n\n"
        )

        outfile.write(
            f"{'Dimension':<30} {' '.join(generation_types):<45} {'Overall':<10}\n"
        )
        outfile.write(f"{'-' * 30} {'-' * 45} {'-' * 10}\n")

        for category, dims in dimensions.items():
            for dim in dims:
                # Calculate standard deviations for each generation type and overall
                sds = [
                    df[df["generation_type"] == gen_type][dim].std()
                    for gen_type in generation_types
                ]
                overall_sd = df[dim].std()

                # Format the line with generation type standard deviations and overall standard deviation
                sd_str = " ".join([f"{sd:.4f}" for sd in sds])
                outfile.write(f"{dim:<30} {sd_str:<45} {overall_sd:.4f}\n")

            # Write category standard deviations
            cat_sds = [
                df[df["generation_type"] == gen_type][dims].values.flatten().std()
                for gen_type in generation_types
            ]
            overall_cat_sd = df[dims].values.flatten().std()

            cat_sd_str = " ".join([f"{sd:.4f}" for sd in cat_sds])
            outfile.write(
                f"{category + ' (Category)':<30} {cat_sd_str:<45} {overall_cat_sd:.4f}\n"
            )
            outfile.write("\n")

        # Section 3: Kruskal-Wallis Test
        outfile.write("2. INFERENTIAL STATISTICS\n")
        outfile.write("=========================\n\n")

        outfile.write("2.1 Kruskal-Wallis H Test Results\n")
        outfile.write("--------------------------------\n\n")

        outfile.write(
            f"{'Dimension':<30} {'H Statistic':<15} {'p-value':<15} {'Significant (p<0.05)':<20}\n"
        )
        outfile.write(f"{'-' * 30} {'-' * 15} {'-' * 15} {'-' * 20}\n")

        for category, dims in dimensions.items():
            for dim in dims:
                # Prepare values for Kruskal-Wallis test
                dim_groups = [
                    df[df["generation_type"] == gen_type][dim].dropna().tolist()
                    for gen_type in generation_types
                ]

                # Calculate H statistic and p-value
                h = calculate_kruskall_wallis(dim_groups)
                p_value = 1 - stats.chi2.cdf(h, len(generation_types) - 1)

                is_significant = "Yes" if p_value < 0.05 else "No"

                outfile.write(
                    f"{dim:<30} {h:.4f}         {p_value:.4f}         {is_significant:<20}\n"
                )

            # Calculate Kruskal-Wallis for category values
            cat_groups = [
                df[df["generation_type"] == gen_type][dims].values.flatten().tolist()
                for gen_type in generation_types
            ]
            h_cat = calculate_kruskall_wallis(cat_groups)
            p_value_cat = 1 - stats.chi2.cdf(h_cat, len(generation_types) - 1)
            is_significant_cat = "Yes" if p_value_cat < 0.05 else "No"

            outfile.write(
                f"{category + ' (Category)':<30} {h_cat:.4f}         {p_value_cat:.4f}         {is_significant_cat:<20}\n"
            )
            outfile.write("\n")

        # Section 4: Effect Size Calculation (Cliff's delta)
        outfile.write("2.2 Effect Size Calculation (Cliff's delta)\n")
        outfile.write("----------------------------------------\n\n")

        # Pairwise combinations of generation types
        gen_type_pairs = [
            (generation_types[i], generation_types[j])
            for i in range(len(generation_types))
            for j in range(i + 1, len(generation_types))
        ]

        outfile.write(
            f"{'Dimension':<30} {'Comparison':<25} {"Cliff's delta":<15} {'Effect Size':<15}\n"
        )
        outfile.write(f"{'-' * 30} {'-' * 25} {'-' * 15} {'-' * 15}\n")

        for category, dims in dimensions.items():
            for dim in dims:
                # Calculate Cliff's delta for all pairwise comparisons
                for gen_type1, gen_type2 in gen_type_pairs:
                    group1 = (
                        df[df["generation_type"] == gen_type1][dim].dropna().tolist()
                    )
                    group2 = (
                        df[df["generation_type"] == gen_type2][dim].dropna().tolist()
                    )

                    cliff_delta = calculate_cliffs_delta(group1, group2)

                    # Interpret effect size
                    def interpret_effect_size(delta):
                        abs_delta = abs(delta)
                        if abs_delta < 0.147:
                            return "Negligible"
                        elif abs_delta < 0.33:
                            return "Small"
                        elif abs_delta < 0.474:
                            return "Medium"
                        else:
                            return "Large"

                    comparison = f"{gen_type1} vs {gen_type2}"
                    outfile.write(
                        f"{dim:<30} {comparison:<25} {cliff_delta:.4f}         {interpret_effect_size(cliff_delta):<15}\n"
                    )

            outfile.write("\n")

        # Section 5: Distribution analysis
        outfile.write("2.3 Distribution analysis of scores across dimensions\n")
        outfile.write("--------------------------------------------------\n\n")

        outfile.write(
            f"{'Category':<20} {'Score 1 (%)':<15} {'Score 2 (%)':<15} {'Score 3 (%)':<15} {'Score 4 (%)':<15} {'Score 5 (%)':<15} {'Median':<10} {'IQR':<10}\n"
        )
        outfile.write(
            f"{'-' * 20} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 10} {'-' * 10}\n"
        )

        # Calculate distribution for each dimension
        for dim in all_dimensions:
            values = df[dim].dropna()
            count = len(values)

            # Calculate frequencies
            score_counts = {i: sum(values == i) for i in range(1, 6)}

            # Calculate percentages
            score_percentages = {
                score: (count / len(values)) * 100
                for score, count in score_counts.items()
            }

            # Calculate median and IQR
            sorted_values = sorted(values)
            median = np.median(sorted_values)
            q1 = np.percentile(sorted_values, 25)
            q3 = np.percentile(sorted_values, 75)
            iqr = q3 - q1

            outfile.write(
                f"{dim:<20} {score_percentages.get(1, 0):.2f}%         {score_percentages.get(2, 0):.2f}%         {score_percentages.get(3, 0):.2f}%         {score_percentages.get(4, 0):.2f}%         {score_percentages.get(5, 0):.2f}%         {median:<10} {iqr:<10}\n"
            )

        # Section 6: Confidence Intervals
        outfile.write("\n\n2.4 Confidence Intervals (95%)\n")
        outfile.write("-----------------------------\n\n")

        outfile.write(
            f"{'Dimension':<30} {'Approach':<15} {'Mean':<10} {'95% CI Lower':<15} {'95% CI Upper':<15}\n"
        )
        outfile.write(f"{'-' * 30} {'-' * 15} {'-' * 10} {'-' * 15} {'-' * 15}\n")

        for dim in all_dimensions:
            for gen_type in generation_types:
                values = df[df["generation_type"] == gen_type][dim].dropna()

                if len(values) > 0:
                    mean = values.mean()
                    std_err = values.std() / math.sqrt(len(values))
                    ci_lower = mean - 1.96 * std_err
                    ci_upper = mean + 1.96 * std_err

                    approach = gen_type.capitalize()

                    outfile.write(
                        f"{dim:<30} {approach:<15} {mean:.4f}    {ci_lower:.4f}         {ci_upper:.4f}\n"
                    )

        # Section 7: LLM-Specific Analysis
        outfile.write("\n5. LLM-SPECIFIC ANALYSIS\n")
        outfile.write("=======================\n\n")

        # Overall mean scores per LLM
        outfile.write("5.1 Mean Scores by LLM\n")
        outfile.write("--------------------\n\n")
        outfile.write(f"{'LLM':<20}")
        for category in dimensions.keys():
            outfile.write(f"{category:<20}")
        outfile.write("Overall\n")
        outfile.write("-" * (20 * (len(dimensions) + 1)) + "\n")

        for llm in llms:
            llm_df = df[df["id"] == llm]
            outfile.write(f"{llm:<20}")

            for category, dims in dimensions.items():
                cat_mean = llm_df[dims].values.flatten().mean()
                outfile.write(f"{cat_mean:.4f}{' ' * (20 - len(str(cat_mean)))}")

            overall_mean = llm_df[all_dimensions].values.flatten().mean()
            outfile.write(f"{overall_mean:.4f}\n")

        # High Variability Analysis
        outfile.write("\n5.2 Variability Analysis by LLM\n")
        outfile.write("------------------------------\n\n")
        outfile.write(f"{'LLM':<20}")
        for category in dimensions.keys():
            outfile.write(f"{category:<20}")
        outfile.write("Overall Std Dev\n")
        outfile.write("-" * (20 * (len(dimensions) + 1)) + "\n")

        for llm in llms:
            llm_df = df[df["id"] == llm]
            outfile.write(f"{llm:<20}")

            for category, dims in dimensions.items():
                cat_std = llm_df[dims].values.flatten().std()
                outfile.write(f"{cat_std:.4f}{' ' * (20 - len(str(cat_std)))}")

            overall_std = llm_df[all_dimensions].values.flatten().std()
            outfile.write(f"{overall_std:.4f}\n")

        # Detailed Category Breakdown
        outfile.write("\n5.3 Detailed Category Breakdown\n")
        outfile.write("-------------------------------\n\n")

        for category, dims in dimensions.items():
            outfile.write(f"\n{category} Detailed Breakdown:\n")
            outfile.write("-" * (len(category) + 11) + "\n")

            outfile.write(f"{'Dimension':<25}")
            for llm in llms:
                outfile.write(f"{llm:<15}")
            outfile.write("Overall\n")
            outfile.write("-" * (25 + 15 * len(llms) + 10) + "\n")

            for dim in dims:
                outfile.write(f"{dim:<25}")
                for llm in llms:
                    llm_mean = df[df["id"] == llm][dim].mean()
                    outfile.write(f"{llm_mean:.4f}{' ' * (15 - len(str(llm_mean)))}")

                overall_mean = df[dim].mean()
                outfile.write(f"{overall_mean:.4f}\n")

        # Clean up and wrap up the analysis
        print("Analysis complete! Results written to llms_results.txt")


if __name__ == "__main__":
    main()
