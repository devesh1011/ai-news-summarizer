import os
import pandas as pd

# Define paths
news_articles_path = "Epiphany/data/News Articles"
summaries_path = "Epiphany/data/Summaries"

# Prepare a list to hold data
data = []

# Iterate through categories (subfolders)
for category in os.listdir(news_articles_path):
    # Paths to article and summary subfolders
    article_folder = os.path.join(news_articles_path, category)
    summary_folder = os.path.join(summaries_path, category)

    if os.path.isdir(article_folder) and os.path.isdir(summary_folder):
        # Iterate through article files
        for article_file in os.listdir(article_folder):
            if article_file.endswith(".txt"):
                # Paths to article and corresponding summary
                article_path = os.path.join(article_folder, article_file)
                summary_path = os.path.join(summary_folder, article_file)

                # Ensure corresponding summary exists
                if os.path.exists(summary_path):
                    try:
                        # Read article
                        with open(article_path, "r", encoding="utf-8") as af:
                            article = af.read().strip()
                    except UnicodeDecodeError:
                        print(f"Skipping {article_path} due to encoding issues.")
                        continue

                    try:
                        # Read summary
                        with open(summary_path, "r", encoding="utf-8") as sf:
                            summary = sf.read().strip()
                    except UnicodeDecodeError:
                        print(f"Skipping {summary_path} due to encoding issues.")
                        continue

                    # Append to data
                    data.append({"article": article, "summary": summary})

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("news_dataset.csv", index=False)
