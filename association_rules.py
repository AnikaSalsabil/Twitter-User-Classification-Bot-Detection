# Association Rules for Twitter Human vs Bot Classification

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import warnings
import os

warnings.filterwarnings("ignore")

# ============================================
# STEP 1: Load and Explore Data
# ============================================

# Load your CSV file - update the path to your actual file
df = pd.read_csv('out/df_cleaned.csv')  # UPDATE THIS PATH TO YOUR CSV FILE

FIG_DIR = "out"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


def savefile(obj, name, file_type='csv'):
    """
    Save a file (CSV or text) to the FIG_DIR directory.

    Parameters:
    - obj: The object to save (DataFrame for CSV, string for text)
    - name: The filename (including extension, e.g., 'file.csv' or 'file.txt')
    - file_type: Type of file ('csv' or 'txt')
    """
    path = os.path.join(FIG_DIR, name)
    if file_type == 'csv':
        obj.to_csv(path, index=False)
    elif file_type == 'txt':
        with open(path, 'w') as f:
            f.write(obj)
    else:
        raise ValueError("Unsupported file_type. Use 'csv' or 'txt'.")
    print(f"[Saved] {path}")

# Display basic information about the dataset
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Dataset Shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Display unique values for key columns
print("\n" + "=" * 60)
print("UNIQUE VALUES IN KEY COLUMNS")
print("=" * 60)
print(f"Unique genders: {df['gender'].unique()}")
print(f"Gender value counts:\n{df['gender'].value_counts()}")
print(f"\n_golden values: {df['_golden'].unique()}")
print(f"_trusted_judgments range: {df['_trusted_judgments'].min()} to {df['_trusted_judgments'].max()}")


# ============================================
# STEP 2: Feature Engineering for Association Rules
# ============================================

def prepare_twitter_features(df):
    """
    Prepare Twitter data for association rule mining
    Creates binary features from the available columns
    """

    # Initialize empty DataFrame for processed features
    df_processed = pd.DataFrame()

    # 1. Process gender column (assuming 'brand' might be bots)
    if 'gender' in df.columns:
        df_processed['is_male'] = (df['gender'] == 'male').astype(int)
        df_processed['is_female'] = (df['gender'] == 'female').astype(int)
        df_processed['is_brand'] = (df['gender'] == 'brand').astype(int)

    # 2. Process gender confidence
    if 'gender:confidence' in df.columns:
        # Create bins for confidence levels
        df_processed['high_confidence'] = (df['gender:confidence'] >= 0.8).astype(int)
        df_processed['medium_confidence'] = ((df['gender:confidence'] >= 0.5) &
                                             (df['gender:confidence'] < 0.8)).astype(int)
        df_processed['low_confidence'] = (df['gender:confidence'] < 0.5).astype(int)

    # 3. Process description field
    if 'description' in df.columns:
        # Check for missing descriptions
        df_processed['has_description'] = (~df['description'].isna() &
                                           (df['description'] != 'missing_description')).astype(int)
        df_processed['no_description'] = (df['description'].isna() |
                                          (df['description'] == 'missing_description')).astype(int)

        # Extract features from description text (when not missing)
        desc_text = df['description'].fillna('')

        # Common bot/spam indicators in description
        df_processed['desc_has_url'] = desc_text.str.contains('http|www', case=False, na=False).astype(int)
        df_processed['desc_has_emoji'] = desc_text.str.contains('Ã', na=False).astype(int)
        df_processed['desc_very_short'] = (desc_text.str.len() < 20).astype(int)
        df_processed['desc_very_long'] = (desc_text.str.len() > 150).astype(int)

        # Common human indicators
        df_processed['desc_has_job_title'] = desc_text.str.contains(
            'teacher|student|mom|dad|wife|husband|engineer|writer|artist',
            case=False, na=False).astype(int)
        df_processed['desc_has_location'] = desc_text.str.contains(
            'university|college|school|from|living',
            case=False, na=False).astype(int)

    # 4. Process text field (tweet content)
    if 'text' in df.columns:
        tweet_text = df['text'].fillna('')

        # Tweet characteristics
        df_processed['tweet_has_url'] = tweet_text.str.contains('https://t.co', na=False).astype(int)
        df_processed['tweet_has_hashtag'] = tweet_text.str.contains('#', na=False).astype(int)
        df_processed['tweet_has_mention'] = tweet_text.str.contains('@', na=False).astype(int)
        df_processed['tweet_very_short'] = (tweet_text.str.len() < 50).astype(int)
        df_processed['tweet_long'] = (tweet_text.str.len() > 200).astype(int)

        # Check for retweet patterns
        df_processed['is_retweet'] = tweet_text.str.contains('RT @', na=False).astype(int)

        # Check for special characters (potential spam)
        df_processed['has_special_chars'] = tweet_text.str.contains('Ã|Â|_Ã™', na=False).astype(int)

    # 5. Process _golden field (if relevant)
    if '_golden' in df.columns:
        df_processed['is_golden'] = (df['_golden'] == True).astype(int)

    # 6. Process _trusted_judgments
    if '_trusted_judgments' in df.columns:
        df_processed['high_trust'] = (df['_trusted_judgments'] >= 3).astype(int)
        df_processed['low_trust'] = (df['_trusted_judgments'] < 3).astype(int)

    # 7. Create target variable based on heuristics
    # Since we don't have explicit labels, we'll use heuristics:
    # Brands are likely bots, high confidence humans are likely human
    df_processed['likely_human'] = (
        ((df['gender'].isin(['male', 'female'])) &
         (df['gender:confidence'] >= 0.8) &
         (~df['description'].isna()) &
         (df['description'] != 'missing_description'))
    ).astype(int)

    df_processed['likely_bot'] = (
            (df['gender'] == 'brand') |
            (df['gender:confidence'] < 0.5) |
            (df['description'].isna()) |
            (df['description'] == 'missing_description')
    ).astype(int)

    return df_processed


# Prepare the data
print("\n" + "=" * 60)
print("PREPARING DATA FOR ASSOCIATION RULES")
print("=" * 60)
df_prepared = prepare_twitter_features(df)

print(f"Prepared data shape: {df_prepared.shape}")
print(f"Prepared features: {df_prepared.columns.tolist()}")
print(f"\nFeature statistics:")
print(df_prepared.sum().sort_values(ascending=False))

# ============================================
# STEP 3: Generate Frequent Itemsets
# ============================================

print("\n" + "=" * 60)
print("GENERATING FREQUENT ITEMSETS")
print("=" * 60)

# Generate frequent itemsets with different support levels
support_levels = [0.1, 0.05, 0.03]
best_itemsets = None
best_support = None

for min_sup in support_levels:
    try:
        frequent_itemsets = apriori(df_prepared,
                                    min_support=min_sup,
                                    use_colnames=True)

        if len(frequent_itemsets) >= 10:  # We want at least 10 itemsets
            best_itemsets = frequent_itemsets
            best_support = min_sup
            print(f"✓ Using min_support={min_sup}: Found {len(frequent_itemsets)} itemsets")
            break
        else:
            print(f"  min_support={min_sup}: Only {len(frequent_itemsets)} itemsets (too few)")
    except:
        print(f"  min_support={min_sup}: Failed to generate itemsets")

if best_itemsets is None:
    print("Warning: Could not generate sufficient itemsets. Using min_support=0.01")
    frequent_itemsets = apriori(df_prepared, min_support=0.01, use_colnames=True)
else:
    frequent_itemsets = best_itemsets

print(f"\nTop 10 frequent itemsets by support:")
if len(frequent_itemsets) > 0:
    print(frequent_itemsets.nlargest(10, 'support')[['itemsets', 'support']])

# ============================================
# STEP 4: Generate Association Rules
# ============================================

print("\n" + "=" * 60)
print("GENERATING ASSOCIATION RULES")
print("=" * 60)

if len(frequent_itemsets) > 0:
    # Generate rules with lift threshold
    rules = association_rules(frequent_itemsets,
                              metric="lift",
                              min_threshold=1.0)

    print(f"Total rules generated: {len(rules)}")

    if len(rules) > 0:
        # ============================================
        # STEP 5: Analyze Rules for Human vs Bot Classification
        # ============================================

        print("\n" + "=" * 60)
        print("RULES FOR HUMAN CLASSIFICATION")
        print("=" * 60)

        # Find rules that predict human accounts
        human_rules = rules[rules['consequents'].apply(
            lambda x: 'likely_human' in str(x) or 'is_male' in str(x) or 'is_female' in str(x)
        )]

        if len(human_rules) > 0:
            print(f"Found {len(human_rules)} rules suggesting human accounts")
            print("\nTop 5 HUMAN-predicting rules (by confidence * lift):")
            human_rules['score'] = human_rules['confidence'] * human_rules['lift']
            top_human = human_rules.nlargest(5, 'score')[
                ['antecedents', 'consequents', 'support', 'confidence', 'lift']
            ]
            for idx, row in top_human.iterrows():
                print(f"\nRule: {row['antecedents']} => {row['consequents']}")
                print(f"  Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f}")

        print("\n" + "=" * 60)
        print("RULES FOR BOT CLASSIFICATION")
        print("=" * 60)

        # Find rules that predict bot accounts
        bot_rules = rules[rules['consequents'].apply(
            lambda x: 'likely_bot' in str(x) or 'is_brand' in str(x)
        )]

        if len(bot_rules) > 0:
            print(f"Found {len(bot_rules)} rules suggesting bot accounts")
            print("\nTop 5 BOT-predicting rules (by confidence * lift):")
            bot_rules['score'] = bot_rules['confidence'] * bot_rules['lift']
            top_bot = bot_rules.nlargest(5, 'score')[
                ['antecedents', 'consequents', 'support', 'confidence', 'lift']
            ]
            for idx, row in top_bot.iterrows():
                print(f"\nRule: {row['antecedents']} => {row['consequents']}")
                print(f"  Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f}")

        # ============================================
        # STEP 6: High Quality Rules Analysis
        # ============================================

        print("\n" + "=" * 60)
        print("HIGH QUALITY RULES")
        print("=" * 60)

        # Filter for high confidence and high lift
        high_quality_rules = rules[(rules['confidence'] >= 0.6) & (rules['lift'] >= 1.2)]

        if len(high_quality_rules) > 0:
            print(f"Found {len(high_quality_rules)} high quality rules (confidence >= 0.6, lift >= 1.2)")
            print("\nTop 10 high quality rules:")
            high_quality_rules['score'] = high_quality_rules['confidence'] * high_quality_rules['lift']
            top_quality = high_quality_rules.nlargest(10, 'score')[
                ['antecedents', 'consequents', 'support', 'confidence', 'lift']
            ]
            for idx, row in top_quality.iterrows():
                print(f"\nRule: {row['antecedents']} => {row['consequents']}")
                print(f"  Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f}")

        # ============================================
        # STEP 7: Interesting Pattern Discovery
        # ============================================

        print("\n" + "=" * 60)
        print("INTERESTING PATTERNS")
        print("=" * 60)

        # Find rules with description features
        desc_rules = rules[rules['antecedents'].apply(
            lambda x: any('desc' in str(item) for item in x)
        )]

        if len(desc_rules) > 0:
            print(f"\nDescription-based rules: {len(desc_rules)}")
            print("Top 3 description patterns:")
            for idx, row in desc_rules.nlargest(3, 'lift').iterrows():
                print(f"  {row['antecedents']} => {row['consequents']} (Lift: {row['lift']:.2f})")

        # Find rules with tweet content features
        tweet_rules = rules[rules['antecedents'].apply(
            lambda x: any('tweet' in str(item) for item in x)
        )]

        if len(tweet_rules) > 0:
            print(f"\nTweet content-based rules: {len(tweet_rules)}")
            print("Top 3 tweet patterns:")
            for idx, row in tweet_rules.nlargest(3, 'lift').iterrows():
                print(f"  {row['antecedents']} => {row['consequents']} (Lift: {row['lift']:.2f})")

        # ============================================
        # STEP 8: Visualization
        # ============================================

        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATIONS")
        print("=" * 60)

        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Define lift categories for legend clarity
        lift_bins = [1, 2, 3, 4, 5, rules['lift'].max()]
        lift_labels = [f"{lift_bins[i]}–{lift_bins[i + 1]}" for i in range(len(lift_bins) - 1)]
        rules['lift_bin'] = pd.cut(rules['lift'], bins=lift_bins, labels=lift_labels, include_lowest=True)



        # 1. Scatter plot of rules
        scatter = axes[0, 0].scatter(rules['support'], rules['confidence'],
                                     c=rules['lift'], s=50, alpha=0.6, cmap='viridis')

        """
        # Scatter plot with legend
        for label, group in rules.groupby('lift_bin'):
            axes[0, 0].scatter(group['support'], group['confidence'],
                               s=50, alpha=0.6, label=f"Lift {label}")
        """

        axes[0, 0].set_xlabel('Support')
        axes[0, 0].set_ylabel('Confidence')
        axes[0, 0].set_title('Association Rules: Support vs Confidence (colored by Lift)')
        plt.colorbar(scatter, ax=axes[0, 0])
        axes[0, 0].grid(True, alpha=0.3)


        # 2. Lift distribution
        axes[0, 1].hist(rules['lift'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Lift')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Lift Values')
        axes[0, 1].axvline(x=1, color='red', linestyle='--', label='Lift=1')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Confidence distribution
        axes[1, 0].hist(rules['confidence'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Confidence Values')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Top features by frequency
        feature_counts = df_prepared.sum().sort_values(ascending=False).head(10)
        axes[1, 1].barh(range(len(feature_counts)), feature_counts.values)
        axes[1, 1].set_yticks(range(len(feature_counts)))
        axes[1, 1].set_yticklabels(feature_counts.index)
        axes[1, 1].set_xlabel('Count')
        axes[1, 1].set_title('Top 10 Most Frequent Features')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        #plt.savefig('twitter_association_rules_analysis.png', dpi=300, bbox_inches='tight')
        savefig('twitter_association_rules_analysis.png')

        #plt.show()

        print("Visualizations saved as 'twitter_association_rules_analysis.png'")

        # ================================
        # NEW: Correlation Heatmap
        # ================================
        import seaborn as sns

        corr = rules[['support', 'confidence', 'lift']].corr()

        plt.figure(figsize=(6, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation between Support, Confidence, and Lift")
        plt.tight_layout()
        #plt.savefig('twitter_rules_correlation.png', dpi=300, bbox_inches='tight')

        savefig('twitter_rules_correlation.png')

        #plt.show()

        print("Correlation Heatmap saved as 'twitter_rules_correlation.png'")

        # ============================================
        # STEP 9: Export Results
        # ============================================

        print("\n" + "=" * 60)
        print("EXPORTING RESULTS TO /out FOLDER")
        print("=" * 60)

        # Save all rules
        #rules.to_csv('twitter_all_association_rules.csv', index=False)
        savefile(rules, 'twitter_all_association_rules.csv', file_type='csv')
        print("✓ All rules saved to 'twitter_all_association_rules.csv'")

        # Save human-indicating rules
        if len(human_rules) > 0:
            #human_rules.to_csv('twitter_human_rules.csv', index=False)
            savefile(human_rules, 'twitter_human_rules.csv', file_type='csv')
            print(f"✓ {len(human_rules)} human rules saved to 'twitter_human_rules.csv'")

        # Save bot-indicating rules
        if len(bot_rules) > 0:
            #bot_rules.to_csv('twitter_bot_rules.csv', index=False)
            savefile(bot_rules, 'twitter_bot_rules.csv', file_type='csv')
            print(f"✓ {len(bot_rules)} bot rules saved to 'twitter_bot_rules.csv'")

        # Save high quality rules
        if len(high_quality_rules) > 0:
            #high_quality_rules.to_csv('twitter_high_quality_rules.csv', index=False)
            savefile(high_quality_rules, 'twitter_high_quality_rules.csv', file_type='csv')
            print(f"✓ {len(high_quality_rules)} high quality rules saved to 'twitter_high_quality_rules.csv'")

        # Save summary report
        summary_text = (
                "TWITTER BOT DETECTION - ASSOCIATION RULES ANALYSIS SUMMARY\n"
                + "=" * 60 + "\n\n"
                + f"Dataset shape: {df.shape}\n"
                + f"Prepared features: {df_prepared.shape[1]}\n"
                + f"Min support used: {best_support if best_support else 0.01}\n"
                + f"Total itemsets found: {len(frequent_itemsets)}\n"
                + f"Total rules generated: {len(rules)}\n"
                + f"Human-indicating rules: {len(human_rules)}\n"
                + f"Bot-indicating rules: {len(bot_rules)}\n"
                + f"High quality rules: {len(high_quality_rules)}\n"
                + f"\nTop rule by lift: {rules.nlargest(1, 'lift')['antecedents'].values[0]} => {rules.nlargest(1, 'lift')['consequents'].values[0]}\n"
                + f"Lift: {rules.nlargest(1, 'lift')['lift'].values[0]:.3f}\n"
        )
        savefile(summary_text, 'twitter_association_rules_summary.txt', file_type='txt')
        print("✓ Summary saved to 'twitter_association_rules_summary.txt'")

    else:
        print("No rules could be generated. Try adjusting the min_support or min_threshold values.")
else:
    print("No frequent itemsets found. The data might be too sparse for association rules.")
    print("Consider:")
    print("1. Lowering the min_support value further")
    print("2. Creating fewer, more general features")
    print("3. Checking if the data has enough patterns for association rule mining")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)