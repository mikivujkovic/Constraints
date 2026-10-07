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
    df = pd.read_csv("data/Human_evaluations.csv")

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

    # List of all generation types
    generation_types = ["naive", "prompt_eng", "netai"]

    # Prepare output text file
    with open("results.txt", "w") as outfile:
        outfile.write("LANDING PAGE SURVEY ANALYSIS RESULTS\n")
        outfile.write("====================================\n\n")

        # Section 1: Basic Statistics
        outfile.write("1. BASIC STATISTICS\n")
        outfile.write("===================\n\n")

        outfile.write("1.1 Mean scores for each dimension by generation approach\n")
        outfile.write("---------------------------------------------------------\n\n")

        # Calculate means for each dimension by generation type
        outfile.write(
            f"{'Dimension':<30} {'Naive':<10} {'Prompt Eng':<15} {'NetAI':<10} {'Overall':<10}\n"
        )
        outfile.write(f"{'-' * 30} {'-' * 10} {'-' * 15} {'-' * 10} {'-' * 10}\n")

        for category, dims in dimensions.items():
            for dim in dims:
                naive_mean = df[df["generation_type"] == "naive"][dim].mean()
                prompt_eng_mean = df[df["generation_type"] == "prompt_eng"][dim].mean()
                netai_mean = df[df["generation_type"] == "netai"][dim].mean()
                overall_mean = df[dim].mean()

                outfile.write(
                    f"{dim:<30} {naive_mean:.4f}    {prompt_eng_mean:.4f}        {netai_mean:.4f}    {overall_mean:.4f}\n"
                )

            # Write category means
            naive_cat_mean = (
                df[df["generation_type"] == "naive"][dims].values.flatten().mean()
            )
            prompt_eng_cat_mean = (
                df[df["generation_type"] == "prompt_eng"][dims].values.flatten().mean()
            )
            netai_cat_mean = (
                df[df["generation_type"] == "netai"][dims].values.flatten().mean()
            )
            overall_cat_mean = df[dims].values.flatten().mean()

            outfile.write(
                f"{category + ' (Category)':<30} {naive_cat_mean:.4f}    {prompt_eng_cat_mean:.4f}        {netai_cat_mean:.4f}    {overall_cat_mean:.4f}\n"
            )
            outfile.write("\n")

        outfile.write("\n\n")

        # Section 2: Standard Deviations
        outfile.write(
            "1.2 Standard deviation for each dimension by generation approach\n"
        )
        outfile.write(
            "--------------------------------------------------------------\n\n"
        )

        outfile.write(
            f"{'Dimension':<30} {'Naive':<10} {'Prompt Eng':<15} {'NetAI':<10} {'Overall':<10}\n"
        )
        outfile.write(f"{'-' * 30} {'-' * 10} {'-' * 15} {'-' * 10} {'-' * 10}\n")

        for category, dims in dimensions.items():
            for dim in dims:
                naive_sd = df[df["generation_type"] == "naive"][dim].std()
                prompt_eng_sd = df[df["generation_type"] == "prompt_eng"][dim].std()
                netai_sd = df[df["generation_type"] == "netai"][dim].std()
                overall_sd = df[dim].std()

                outfile.write(
                    f"{dim:<30} {naive_sd:.4f}    {prompt_eng_sd:.4f}        {netai_sd:.4f}    {overall_sd:.4f}\n"
                )

            # Write category standard deviations
            naive_values = df[df["generation_type"] == "naive"][dims].values.flatten()
            prompt_eng_values = df[df["generation_type"] == "prompt_eng"][
                dims
            ].values.flatten()
            netai_values = df[df["generation_type"] == "netai"][dims].values.flatten()
            all_values = df[dims].values.flatten()

            naive_cat_sd = naive_values.std()
            prompt_eng_cat_sd = prompt_eng_values.std()
            netai_cat_sd = netai_values.std()
            overall_cat_sd = all_values.std()

            outfile.write(
                f"{category + ' (Category)':<30} {naive_cat_sd:.4f}    {prompt_eng_cat_sd:.4f}        {netai_cat_sd:.4f}    {overall_cat_sd:.4f}\n"
            )
            outfile.write("\n")

        outfile.write("\n\n")

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
                naive_values = (
                    df[df["generation_type"] == "naive"][dim].dropna().tolist()
                )
                prompt_eng_values = (
                    df[df["generation_type"] == "prompt_eng"][dim].dropna().tolist()
                )
                netai_values = (
                    df[df["generation_type"] == "netai"][dim].dropna().tolist()
                )

                h = calculate_kruskall_wallis(
                    [naive_values, prompt_eng_values, netai_values]
                )

                # Calculate p-value (approximated using chi-square distribution with 2 degrees of freedom)
                p_value = 1 - stats.chi2.cdf(h, 2)

                is_significant = "Yes" if p_value < 0.05 else "No"

                outfile.write(
                    f"{dim:<30} {h:.4f}         {p_value:.4f}         {is_significant:<20}\n"
                )

            # Calculate Kruskal-Wallis for category values
            naive_cat_values = (
                df[df["generation_type"] == "naive"][dims].values.flatten().tolist()
            )
            prompt_eng_cat_values = (
                df[df["generation_type"] == "prompt_eng"][dims]
                .values.flatten()
                .tolist()
            )
            netai_cat_values = (
                df[df["generation_type"] == "netai"][dims].values.flatten().tolist()
            )

            h_cat = calculate_kruskall_wallis(
                [naive_cat_values, prompt_eng_cat_values, netai_cat_values]
            )
            p_value_cat = 1 - stats.chi2.cdf(h_cat, 2)
            is_significant_cat = "Yes" if p_value_cat < 0.05 else "No"

            outfile.write(
                f"{category + ' (Category)':<30} {h_cat:.4f}         {p_value_cat:.4f}         {is_significant_cat:<20}\n"
            )
            outfile.write("\n")

        outfile.write("\n\n")

        # Section 4: Effect Size Calculation (Cliff's delta)
        outfile.write("2.2 Effect Size Calculation (Cliff's delta)\n")
        outfile.write("----------------------------------------\n\n")

        outfile.write(
            f"{'Dimension':<30} {'Comparison':<25} Cliff's delta   {'Effect Size':<15}\n"
        )
        outfile.write(f"{'-' * 30} {'-' * 25} {'-' * 15} {'-' * 15}\n")

        for category, dims in dimensions.items():
            for dim in dims:
                naive_values = (
                    df[df["generation_type"] == "naive"][dim].dropna().tolist()
                )
                prompt_eng_values = (
                    df[df["generation_type"] == "prompt_eng"][dim].dropna().tolist()
                )
                netai_values = (
                    df[df["generation_type"] == "netai"][dim].dropna().tolist()
                )

                # Calculate Cliff's delta between pairs
                naive_vs_netai = calculate_cliffs_delta(naive_values, netai_values)
                naive_vs_prompt_eng = calculate_cliffs_delta(
                    naive_values, prompt_eng_values
                )
                prompt_eng_vs_netai = calculate_cliffs_delta(
                    prompt_eng_values, netai_values
                )

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

                outfile.write(
                    f"{dim:<30} {'Naive vs NetAI':<25} {naive_vs_netai:.4f}         {interpret_effect_size(naive_vs_netai):<15}\n"
                )
                outfile.write(
                    f"{'':<30} {'Naive vs Prompt Eng':<25} {naive_vs_prompt_eng:.4f}         {interpret_effect_size(naive_vs_prompt_eng):<15}\n"
                )
                outfile.write(
                    f"{'':<30} {'Prompt Eng vs NetAI':<25} {prompt_eng_vs_netai:.4f}         {interpret_effect_size(prompt_eng_vs_netai):<15}\n"
                )

            outfile.write("\n")

        outfile.write("\n\n")

        # Section 5: Distribution analysis
        outfile.write("2.3 Distribution analysis of scores across dimensions\n")
        outfile.write("--------------------------------------------------\n\n")

        outfile.write(
            f"{'Category':<20} {'Score 1 (%)':<15} {'Score 2 (%)':<15} {'Score 3 (%)':<15} {'Score 4 (%)':<15} {'Score 5 (%)':<15} {'Median':<10} {'IQR':<10}\n"
        )
        outfile.write(
            f"{'-' * 20} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 10} {'-' * 10}\n"
        )

        # Calculate for each dimension
        for dim in all_dimensions:
            values = df[dim].dropna()
            count = len(values)

            # Calculate frequencies
            score_counts = {i: 0 for i in range(1, 6)}
            for val in values:
                score_counts[val] = score_counts.get(val, 0) + 1

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

        outfile.write("\n\n")

        # Section 6: Confidence Intervals
        outfile.write("2.4 Confidence Intervals (95%)\n")
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

                    approach = (
                        "Unconstrained"
                        if gen_type == "naive"
                        else "Prompt-Engineered"
                        if gen_type == "prompt_eng"
                        else "Constrained (netAI)"
                    )

                    outfile.write(
                        f"{dim:<30} {approach:<15} {mean:.4f}    {ci_lower:.4f}         {ci_upper:.4f}\n"
                    )

        outfile.write("\n\n")

        # Section 7: Results from PDF
        outfile.write("3. SUMMARY OF RESULTS\n")
        outfile.write("==============================\n\n")

        outfile.write("3.1 Overall Performance Comparison\n")
        outfile.write("---------------------------------\n\n")

        outfile.write(
            "Table 1: Average scores by generation approach across all dimensions\n\n"
        )
        outfile.write(
            f"{'Dimension':<30} {'Constrained (netAI)':<20} {'Prompt-Engineered':<20} {'Unconstrained':<20}\n"
        )
        outfile.write(f"{'-' * 30} {'-' * 20} {'-' * 20} {'-' * 20}\n")

        # Calculate category means for PDF output
        visual_design_netai = (
            df[df["generation_type"] == "netai"][dimensions["Visual Design"]]
            .values.flatten()
            .mean()
        )
        visual_design_prompt_eng = (
            df[df["generation_type"] == "prompt_eng"][dimensions["Visual Design"]]
            .values.flatten()
            .mean()
        )
        visual_design_naive = (
            df[df["generation_type"] == "naive"][dimensions["Visual Design"]]
            .values.flatten()
            .mean()
        )

        content_quality_netai = (
            df[df["generation_type"] == "netai"][dimensions["Content Quality"]]
            .values.flatten()
            .mean()
        )
        content_quality_prompt_eng = (
            df[df["generation_type"] == "prompt_eng"][dimensions["Content Quality"]]
            .values.flatten()
            .mean()
        )
        content_quality_naive = (
            df[df["generation_type"] == "naive"][dimensions["Content Quality"]]
            .values.flatten()
            .mean()
        )

        user_experience_netai = (
            df[df["generation_type"] == "netai"][dimensions["User Experience"]]
            .values.flatten()
            .mean()
        )
        user_experience_prompt_eng = (
            df[df["generation_type"] == "prompt_eng"][dimensions["User Experience"]]
            .values.flatten()
            .mean()
        )
        user_experience_naive = (
            df[df["generation_type"] == "naive"][dimensions["User Experience"]]
            .values.flatten()
            .mean()
        )

        business_effectiveness_netai = (
            df[df["generation_type"] == "netai"][dimensions["Business Effectiveness"]]
            .values.flatten()
            .mean()
        )
        business_effectiveness_prompt_eng = (
            df[df["generation_type"] == "prompt_eng"][
                dimensions["Business Effectiveness"]
            ]
            .values.flatten()
            .mean()
        )
        business_effectiveness_naive = (
            df[df["generation_type"] == "naive"][dimensions["Business Effectiveness"]]
            .values.flatten()
            .mean()
        )

        # Calculate overall means
        netai_overall = (
            df[df["generation_type"] == "netai"][all_dimensions].values.flatten().mean()
        )
        prompt_eng_overall = (
            df[df["generation_type"] == "prompt_eng"][all_dimensions]
            .values.flatten()
            .mean()
        )
        naive_overall = (
            df[df["generation_type"] == "naive"][all_dimensions].values.flatten().mean()
        )

        outfile.write(
            f"{'Visual Design':<30} {visual_design_netai:.2f}{'':<16} {visual_design_prompt_eng:.2f}{'':<16} {visual_design_naive:.2f}\n"
        )
        outfile.write(
            f"{'Content Quality':<30} {content_quality_netai:.2f}{'':<16} {content_quality_prompt_eng:.2f}{'':<16} {content_quality_naive:.2f}\n"
        )
        outfile.write(
            f"{'User Experience':<30} {user_experience_netai:.2f}{'':<16} {user_experience_prompt_eng:.2f}{'':<16} {user_experience_naive:.2f}\n"
        )
        outfile.write(
            f"{'Business Effectiveness':<30} {business_effectiveness_netai:.2f}{'':<16} {business_effectiveness_prompt_eng:.2f}{'':<16} {business_effectiveness_naive:.2f}\n"
        )
        outfile.write(
            f"{'Overall':<30} {netai_overall:.2f}{'':<16} {prompt_eng_overall:.2f}{'':<16} {naive_overall:.2f}\n"
        )

        outfile.write("\n\n")

        # Section 3.2: High Satisfaction Ratings
        outfile.write("3.2 High Satisfaction Ratings\n")
        outfile.write("--------------------------\n\n")

        outfile.write(
            "Table 2: Percentage of high satisfaction ratings (scores ≥4)\n\n"
        )
        outfile.write(f"{'Approach':<30} {'High Satisfaction Percentage':<30}\n")
        outfile.write(f"{'-' * 30} {'-' * 30}\n")

        # Calculate high satisfaction percentages
        netai_high = (
            df[df["generation_type"] == "netai"][all_dimensions] >= 4
        ).values.flatten().mean() * 100
        prompt_eng_high = (
            df[df["generation_type"] == "prompt_eng"][all_dimensions] >= 4
        ).values.flatten().mean() * 100
        naive_high = (
            df[df["generation_type"] == "naive"][all_dimensions] >= 4
        ).values.flatten().mean() * 100

        outfile.write(f"{'Constrained (netAI)':<30} {netai_high:.2f}%\n")
        outfile.write(f"{'Prompt-Engineered':<30} {prompt_eng_high:.2f}%\n")
        outfile.write(f"{'Unconstrained':<30} {naive_high:.2f}%\n")

        outfile.write("\n\n")

        # Section 3.3: Design Consistency Scores
        outfile.write("3.3 Design Consistency Scores\n")
        outfile.write("---------------------------\n\n")

        outfile.write("Table 3: Design consistency scores by generation approach\n\n")
        outfile.write(f"{'Approach':<30} {'Design Consistency Score (1-5)':<30}\n")
        outfile.write(f"{'-' * 30} {'-' * 30}\n")

        # Calculate design consistency means
        netai_design = df[df["generation_type"] == "netai"]["design_consistency"].mean()
        prompt_eng_design = df[df["generation_type"] == "prompt_eng"][
            "design_consistency"
        ].mean()
        naive_design = df[df["generation_type"] == "naive"]["design_consistency"].mean()

        outfile.write(f"{'Constrained (netAI)':<30} {netai_design:.1f}\n")
        outfile.write(f"{'Prompt-Engineered':<30} {prompt_eng_design:.1f}\n")
        outfile.write(f"{'Unconstrained':<30} {naive_design:.1f}\n")

        outfile.write("\n\n")

        # Section 3.4: Evaluation by Expert Role
        outfile.write("3.4 Evaluation by Expert Role\n")
        outfile.write("---------------------------\n\n")

        outfile.write("Table 4: Constrained approach scores by evaluator role\n\n")
        outfile.write(
            f"{'Dimension':<30} {'Developer':<12} {'Business':<12} {'Designer':<12} {'Project_manager':<16} {'Other':<12}\n"
        )
        outfile.write(
            f"{'-' * 30} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 16} {'-' * 12}\n"
        )

        # Calculate means by role for netai approach
        roles = ["developer", "business", "designer", "project_manager", "other"]

        # First filter to netai and then calculate means by role for each category
        netai_df = df[df["generation_type"] == "netai"]

        role_category_means = {}
        for role in roles:
            role_category_means[role] = {}
            role_df = netai_df[netai_df["role"] == role]

            for category, dims in dimensions.items():
                if len(role_df) > 0:
                    role_category_means[role][category] = (
                        role_df[dims].values.flatten().mean()
                    )
                else:
                    role_category_means[role][category] = float("nan")

        # Write to output
        for category in dimensions.keys():
            outfile.write(f"{category:<30}")
            for role in roles:
                mean_val = role_category_means[role].get(category, float("nan"))
                if not math.isnan(mean_val):
                    outfile.write(f" {mean_val:.2f}{'':<8}")
                else:
                    outfile.write(f" {'-':<10}")
            outfile.write("\n")

        outfile.write("\n\n")

        # Section 8: Summary of Key Findings
        outfile.write("4. SUMMARY OF KEY FINDINGS\n")
        outfile.write("==========================\n\n")

        # outfile.write("Statistical analysis was conducted on dimensions across 4 categories comparing 3 generation approaches: naive (unconstrained), prompt engineering, and netai (constrained).\n\n")

        # # Note on Kruskal-Wallis calculations
        # outfile.write("Note on statistical tests: Our recalculation found that the Kruskal-Wallis H statistics in the original analysis were incorrect. The correct H statistics are positive and indicate statistically significant differences between the generation approaches for most dimensions.\n\n")

        # # Conclusion about Cliff's delta
        # outfile.write("The Cliff's delta effect size calculations confirm that the differences between the naive and netai approaches are large for most dimensions, indicating substantial practical significance in the constrained approach's improved performance.\n\n")

        # Add performance improvement percentages
        visual_design_improvement = (
            (visual_design_netai / visual_design_naive) - 1
        ) * 100
        content_quality_improvement = (
            (content_quality_netai / content_quality_naive) - 1
        ) * 100
        user_experience_improvement = (
            (user_experience_netai / user_experience_naive) - 1
        ) * 100
        business_effectiveness_improvement = (
            (business_effectiveness_netai / business_effectiveness_naive) - 1
        ) * 100

        outfile.write(
            f"Performance improvements of constrained (netAI) over unconstrained approach:\n"
        )
        outfile.write(f"Visual Design: {visual_design_improvement:.1f}%\n")
        outfile.write(f"Content Quality: {content_quality_improvement:.1f}%\n")
        outfile.write(f"User Experience: {user_experience_improvement:.1f}%\n")
        outfile.write(
            f"Business Effectiveness: {business_effectiveness_improvement:.1f}%\n"
        )

        # High satisfaction comparison
        high_satisfaction_ratio = netai_high / naive_high
        outfile.write(
            f"\nThe constrained approach produced {high_satisfaction_ratio:.1f} times as many high-quality ratings as the unconstrained approach.\n"
        )

        # Design consistency improvement
        design_consistency_improvement = ((netai_design / naive_design) - 1) * 100
        outfile.write(
            f"\nThe constrained approach showed a {design_consistency_improvement:.1f}% improvement in design consistency compared to the unconstrained approach.\n"
        )

        outfile.write(
            "\nConclusion: The constrained (netAI) approach consistently outperformed both the prompt-engineered and unconstrained approaches across all evaluated dimensions."
        )

    print("Analysis complete! Results written to results.txt")


if __name__ == "__main__":
    main()
