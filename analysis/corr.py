import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_correlations():
    """
    Calculate correlations between human and LLM evaluations
    """
    
    # Define dimension categories
    dimensions = {
        'Visual Design': [
            'color_harmony_contrast',
            'typography_readability',
            'spatial_layout_balance',
            'visual_hierarchy',
            'design_consistency'
        ],
        'Content Quality': [
            'message_clarity',
            'brand_voice_consistency',
            'cta_effectiveness',
            'value_proposition',
            'content_structure'
        ],
        'User Experience': [
            'navigation_intuitiveness',
            'information_architecture',
            'mobile_responsiveness',
            'load_time_performance',
            'interaction_design'
        ],
        'Business Effectiveness': [
            'conversion_potential',
            'brand_alignment',
            'target_audience',
            'competitive_differentiation',
            'market_readiness'
        ]
    }
    
    # Generation approaches
    generation_types = ['naive', 'prompt_eng', 'netai']
    
    # Human evaluation means (from results.txt)
    human_means = {
        'naive': {
            'Visual Design': 3.4931,
            'Content Quality': 3.5392,
            'User Experience': 3.7085,
            'Business Effectiveness': 3.4115,
            'Overall': 3.54
        },
        'prompt_eng': {
            'Visual Design': 3.6800,
            'Content Quality': 3.6992,
            'User Experience': 3.8131,
            'Business Effectiveness': 3.5792,
            'Overall': 3.69
        },
        'netai': {
            'Visual Design': 4.1523,
            'Content Quality': 4.0954,
            'User Experience': 4.2031,
            'Business Effectiveness': 4.0808,
            'Overall': 4.13
        }
    }
    
    # LLM evaluation means (from llms_results.txt)
    llm_means = {
        'naive': {
            'Visual Design': 3.6125,
            'Content Quality': 4.0250,
            'User Experience': 3.8375,
            'Business Effectiveness': 3.9250,
            'Overall': 3.87
        },
        'prompt_eng': {
            'Visual Design': 3.9125,
            'Content Quality': 4.3000,
            'User Experience': 4.0625,
            'Business Effectiveness': 4.2250,
            'Overall': 4.12
        },
        'netai': {
            'Visual Design': 4.2250,
            'Content Quality': 4.3750,
            'User Experience': 4.1125,
            'Business Effectiveness': 4.5000,
            'Overall': 4.30
        }
    }
    
    # Prepare data for correlation analysis
    human_scores = []
    llm_scores = []
    categories = []
    approaches = []
    
    # Collect scores for each category and approach
    for approach in generation_types:
        for category in ['Visual Design', 'Content Quality', 'User Experience', 'Business Effectiveness']:
            human_scores.append(human_means[approach][category])
            llm_scores.append(llm_means[approach][category])
            categories.append(category)
            approaches.append(approach)
    
    # Also add overall scores
    for approach in generation_types:
        human_scores.append(human_means[approach]['Overall'])
        llm_scores.append(llm_means[approach]['Overall'])
        categories.append('Overall')
        approaches.append(approach)
    
    # Convert to numpy arrays
    human_scores = np.array(human_scores)
    llm_scores = np.array(llm_scores)
    
    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(human_scores, llm_scores)
    spearman_r, spearman_p = stats.spearmanr(human_scores, llm_scores)
    
    # Create results text file
    with open('correlation_analysis.txt', 'w') as f:
        f.write("HUMAN vs LLM CORRELATION ANALYSIS\n")
        f.write("==================================\n\n")
        
        f.write("1. OVERALL CORRELATIONS\n")
        f.write("=======================\n\n")
        
        f.write(f"Pearson correlation coefficient: r = {pearson_r:.4f}\n")
        f.write(f"Pearson p-value: p = {pearson_p:.6f}\n")
        f.write(f"Pearson significance (p<0.01): {'Yes' if float(pearson_p) < 0.01 else 'No'}\n\n")
        
        f.write(f"Spearman correlation coefficient: r = {spearman_r:.4f}\n")
        f.write(f"Spearman p-value: p = {spearman_p:.6f}\n")
        f.write(f"Spearman significance (p<0.01): {'Yes' if float(spearman_p) < 0.01 else 'No'}\n\n")
        
        f.write("2. DETAILED COMPARISON BY CATEGORY\n")
        f.write("==================================\n\n")
        
        f.write(f"{'Category':<25} {'Approach':<15} {'Human Mean':<12} {'LLM Mean':<12} {'Difference':<12}\n")
        f.write(f"{'-'*25} {'-'*15} {'-'*12} {'-'*12} {'-'*12}\n")
        
        for i, (cat, app) in enumerate(zip(categories, approaches)):
            human_score = human_scores[i]
            llm_score = llm_scores[i]
            difference = llm_score - human_score
            
            f.write(f"{cat:<25} {app:<15} {human_score:<12.4f} {llm_score:<12.4f} {difference:<12.4f}\n")
        
        f.write("\n3. PERFORMANCE IMPROVEMENT COMPARISON\n")
        f.write("====================================\n\n")
        
        f.write(f"{'Category':<25} {'Human Improvement':<20} {'LLM Improvement':<18} {'Difference':<12}\n")
        f.write(f"{'-'*25} {'-'*20} {'-'*18} {'-'*12}\n")
        
        # Calculate improvements (netai vs naive)
        for category in ['Visual Design', 'Content Quality', 'User Experience', 'Business Effectiveness', 'Overall']:
            human_improvement = ((human_means['netai'][category] / human_means['naive'][category]) - 1) * 100
            llm_improvement = ((llm_means['netai'][category] / llm_means['naive'][category]) - 1) * 100
            difference = llm_improvement - human_improvement
            
            f.write(f"{category:<25} {human_improvement:<20.1f}% {llm_improvement:<18.1f}% {difference:<12.1f}%\n")
        
        f.write("\n4. RATING INFLATION ANALYSIS\n")
        f.write("============================\n\n")
        
        f.write("Average ratings by evaluator type:\n")
        f.write(f"Human average: {np.mean(human_scores):.4f}\n")
        f.write(f"LLM average: {np.mean(llm_scores):.4f}\n")
        f.write(f"LLM inflation: {((np.mean(llm_scores) / np.mean(human_scores)) - 1) * 100:.1f}%\n\n")
        
        f.write("Standard deviations (measure of differentiation):\n")
        f.write(f"Human std dev: {np.std(human_scores):.4f}\n")
        f.write(f"LLM std dev: {np.std(llm_scores):.4f}\n")
        f.write(f"Differentiation ratio (LLM/Human): {np.std(llm_scores) / np.std(human_scores):.4f}\n")
        
        f.write("\n5. CATEGORY-SPECIFIC CORRELATIONS\n")
        f.write("=================================\n\n")
        
        # Calculate correlations for each category
        categories_unique = ['Visual Design', 'Content Quality', 'User Experience', 'Business Effectiveness']
        
        f.write(f"{'Category':<25} {'Pearson r':<12} {'Pearson p':<12} {'Spearman r':<12} {'Spearman p':<12}\n")
        f.write(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12} {'-'*12}\n")
        
        for category in categories_unique:
            # Get scores for this category across all approaches
            cat_human = [human_means[app][category] for app in generation_types]
            cat_llm = [llm_means[app][category] for app in generation_types]
            
            cat_pearson_r, cat_pearson_p = stats.pearsonr(cat_human, cat_llm)
            cat_spearman_r, cat_spearman_p = stats.spearmanr(cat_human, cat_llm)
            
            f.write(f"{category:<25} {cat_pearson_r:<12.4f} {cat_pearson_p:<12.6f} {cat_spearman_r:<12.4f} {cat_spearman_p:<12.6f}\n")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.subplot(2, 2, 1)
    colors = {'naive': 'red', 'prompt_eng': 'orange', 'netai': 'green'}
    for i, approach in enumerate(approaches):
        mask = np.array(approaches) == approach
        plt.scatter(np.array(human_scores)[mask], np.array(llm_scores)[mask], 
                   c=colors[approach], label=approach.replace('_', ' ').title(), alpha=0.7, s=60)
    
    plt.xlabel('Human Evaluation Scores')
    plt.ylabel('LLM Evaluation Scores')
    plt.title(f'Human vs LLM Evaluations\n(r = {pearson_r:.3f}, p = {pearson_p:.6f})')
    plt.legend()
    
    # Add regression line
    z = np.polyfit(human_scores, llm_scores, 1)
    p = np.poly1d(z)
    plt.plot(human_scores, p(human_scores), "r--", alpha=0.8)
    
    # Add perfect correlation line
    min_val = min(min(human_scores), min(llm_scores))
    max_val = max(max(human_scores), max(llm_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Perfect Correlation')
    
    # Category-wise comparison
    plt.subplot(2, 2, 2)
    categories_for_plot = ['Visual Design', 'Content Quality', 'User Experience', 'Business Effectiveness']
    x_pos = np.arange(len(categories_for_plot))
    
    human_cat_means = [np.mean([human_means[app][cat] for app in generation_types]) for cat in categories_for_plot]
    llm_cat_means = [np.mean([llm_means[app][cat] for app in generation_types]) for cat in categories_for_plot]
    
    width = 0.35
    plt.bar(x_pos - width/2, human_cat_means, width, label='Human', alpha=0.8)
    plt.bar(x_pos + width/2, llm_cat_means, width, label='LLM', alpha=0.8)
    
    plt.xlabel('Categories')
    plt.ylabel('Average Score')
    plt.title('Average Scores by Category')
    plt.xticks(x_pos, [cat.replace(' ', '\n') for cat in categories_for_plot])
    plt.legend()
    
    # Improvement comparison
    plt.subplot(2, 2, 3)
    human_improvements = [((human_means['netai'][cat] / human_means['naive'][cat]) - 1) * 100 
                         for cat in categories_for_plot]
    llm_improvements = [((llm_means['netai'][cat] / llm_means['naive'][cat]) - 1) * 100 
                       for cat in categories_for_plot]
    
    plt.bar(x_pos - width/2, human_improvements, width, label='Human', alpha=0.8)
    plt.bar(x_pos + width/2, llm_improvements, width, label='LLM', alpha=0.8)
    
    plt.xlabel('Categories')
    plt.ylabel('Improvement (%)')
    plt.title('Performance Improvement\n(Constrained vs Unconstrained)')
    plt.xticks(x_pos, [cat.replace(' ', '\n') for cat in categories_for_plot])
    plt.legend()
    
    # Approach comparison
    plt.subplot(2, 2, 4)
    approach_labels = ['Naive', 'Prompt Eng', 'NetAI']
    human_approach_means = [human_means[app]['Overall'] for app in generation_types]
    llm_approach_means = [llm_means[app]['Overall'] for app in generation_types]
    
    x_pos = np.arange(len(approach_labels))
    plt.bar(x_pos - width/2, human_approach_means, width, label='Human', alpha=0.8)
    plt.bar(x_pos + width/2, llm_approach_means, width, label='LLM', alpha=0.8)
    
    plt.xlabel('Approach')
    plt.ylabel('Overall Score')
    plt.title('Overall Scores by Approach')
    plt.xticks(x_pos, approach_labels)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('human_llm_correlation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Analysis complete!")
    print(f"Overall correlation (Pearson): r = {pearson_r:.4f}, p = {pearson_p:.6f}")
    print(f"Overall correlation (Spearman): r = {spearman_r:.4f}, p = {spearman_p:.6f}")
    print(f"Results saved to: correlation_analysis.txt")
    print(f"Visualization saved to: human_llm_correlation.png")
    
    return pearson_r, pearson_p, spearman_r, spearman_p

if __name__ == "__main__":
    calculate_correlations()